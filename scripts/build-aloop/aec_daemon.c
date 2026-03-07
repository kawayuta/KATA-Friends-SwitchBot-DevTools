/*
 * aec_daemon.c — AEC (Acoustic Echo Cancellation) daemon using speexdsp
 *
 * 6-channel mode: processes hw:0,0 (6ch) mic input, applies AEC to 4 mic
 * channels individually, and outputs 6ch to loopback for media/pet_voice.
 *
 * Architecture:
 *   Reference thread: hw:1,1,0 (loopback cap, TTS mono ref) → ring buffer + speaker
 *   Main thread:      cap_dsnoop (6ch mic) + ring buffer → speex AEC (×4) → hw:1,0,1 (6ch loopback)
 *
 * Channel mapping (from RK VQE config, ref_pos: 1):
 *   ch0: mic0 (AEC)
 *   ch1: speaker reference (passthrough)
 *   ch2: mic1 (AEC)
 *   ch3: mic2 (AEC)
 *   ch4: mic3 (AEC)
 *   ch5: unknown (passthrough)
 *
 * Parameters: 16kHz, S16_LE, 160 samples/frame (10ms), 2048 filter length (128ms)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <alsa/asoundlib.h>
#include <speex/speex_echo.h>
#include <speex/speex_preprocess.h>

#define SAMPLE_RATE   16000
#define MIC_CHANNELS  6             /* 6ch from hw:0,0 */
#define REF_CHANNELS  1             /* mono TTS reference */
#define FRAME_SAMPLES 160           /* 10ms at 16kHz per channel */
#define FILTER_LENGTH 2048          /* 128ms tail */
#define RING_FRAMES   256           /* ring buffer capacity in frames */
#define FRAME_BYTES_MONO (FRAME_SAMPLES * sizeof(int16_t))

/* Indices of mic channels to apply AEC (4 mics) */
#define NUM_AEC_CHANNELS 4
static const int AEC_CH[NUM_AEC_CHANNELS] = {0, 2, 3, 4};
/* Indices of passthrough channels */
#define NUM_PASS_CHANNELS 2
static const int PASS_CH[NUM_PASS_CHANNELS] = {1, 5};

static volatile sig_atomic_t g_running = 1;

/* --- Ring buffer (single-producer, single-consumer, mono frames) --- */
typedef struct {
    int16_t buf[RING_FRAMES * FRAME_SAMPLES];
    unsigned int head;  /* write position (frames) */
    unsigned int tail;  /* read position (frames) */
    pthread_mutex_t mtx;
} RingBuf;

static void ring_init(RingBuf *r) {
    memset(r, 0, sizeof(*r));
    pthread_mutex_init(&r->mtx, NULL);
}

static void ring_write(RingBuf *r, const int16_t *frame) {
    pthread_mutex_lock(&r->mtx);
    unsigned int idx = r->head % RING_FRAMES;
    memcpy(&r->buf[idx * FRAME_SAMPLES], frame, FRAME_BYTES_MONO);
    r->head++;
    if (r->head - r->tail > RING_FRAMES)
        r->tail = r->head - RING_FRAMES;
    pthread_mutex_unlock(&r->mtx);
}

static int ring_read(RingBuf *r, int16_t *frame) {
    pthread_mutex_lock(&r->mtx);
    if (r->tail == r->head) {
        pthread_mutex_unlock(&r->mtx);
        return 0;
    }
    unsigned int idx = r->tail % RING_FRAMES;
    memcpy(frame, &r->buf[idx * FRAME_SAMPLES], FRAME_BYTES_MONO);
    r->tail++;
    pthread_mutex_unlock(&r->mtx);
    return 1;
}

static RingBuf g_ring;

/* --- ALSA helpers --- */
static snd_pcm_t *open_alsa(const char *dev, snd_pcm_stream_t dir, unsigned int channels) {
    snd_pcm_t *pcm = NULL;
    int err;

    if ((err = snd_pcm_open(&pcm, dev, dir, 0)) < 0) {
        fprintf(stderr, "Cannot open %s (%s): %s\n",
                dev, dir == SND_PCM_STREAM_CAPTURE ? "capture" : "playback",
                snd_strerror(err));
        return NULL;
    }

    snd_pcm_hw_params_t *params;
    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(pcm, params);

    snd_pcm_hw_params_set_access(pcm, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm, params, channels);

    unsigned int rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(pcm, params, &rate, 0);

    snd_pcm_uframes_t period = FRAME_SAMPLES;
    snd_pcm_hw_params_set_period_size_near(pcm, params, &period, 0);

    snd_pcm_uframes_t buffer_size = FRAME_SAMPLES * 8;
    snd_pcm_hw_params_set_buffer_size_near(pcm, params, &buffer_size);

    if ((err = snd_pcm_hw_params(pcm, params)) < 0) {
        fprintf(stderr, "HW params error on %s: %s\n", dev, snd_strerror(err));
        snd_pcm_close(pcm);
        return NULL;
    }

    /* Log actual parameters */
    unsigned int actual_ch;
    snd_pcm_hw_params_get_channels(params, &actual_ch);
    unsigned int actual_rate;
    snd_pcm_hw_params_get_rate(params, &actual_rate, 0);
    printf("  opened %s: %uch, %uHz\n", dev, actual_ch, actual_rate);

    return pcm;
}

static int alsa_recover(snd_pcm_t *pcm, int err) {
    if (err == -EPIPE) {
        snd_pcm_prepare(pcm);
        return 0;
    } else if (err == -ESTRPIPE) {
        while ((err = snd_pcm_resume(pcm)) == -EAGAIN)
            usleep(10000);
        if (err < 0)
            snd_pcm_prepare(pcm);
        return 0;
    }
    return err;
}

/* --- Reference thread: capture TTS from loopback (mono), feed ring buf + speaker --- */
static void *ref_thread(void *arg) {
    (void)arg;

    snd_pcm_t *cap = open_alsa("hw:1,1,0", SND_PCM_STREAM_CAPTURE, REF_CHANNELS);
    snd_pcm_t *play = open_alsa("softvol_ply", SND_PCM_STREAM_PLAYBACK, REF_CHANNELS);

    if (!cap || !play) {
        fprintf(stderr, "ref_thread: failed to open devices\n");
        g_running = 0;
        return NULL;
    }

    int16_t frame[FRAME_SAMPLES];

    while (g_running) {
        snd_pcm_sframes_t n = snd_pcm_readi(cap, frame, FRAME_SAMPLES);
        if (n < 0) {
            if (alsa_recover(cap, n) < 0) break;
            continue;
        }
        if (n != FRAME_SAMPLES) continue;

        ring_write(&g_ring, frame);

        snd_pcm_sframes_t w = snd_pcm_writei(play, frame, FRAME_SAMPLES);
        if (w < 0) {
            if (alsa_recover(play, w) < 0) break;
        }
    }

    snd_pcm_close(cap);
    snd_pcm_close(play);
    return NULL;
}

/* --- Deinterleave: 6ch interleaved → separate mono channels --- */
static void deinterleave(const int16_t *interleaved, int16_t ch_buf[][FRAME_SAMPLES],
                         int num_channels, int frame_samples) {
    for (int s = 0; s < frame_samples; s++) {
        for (int c = 0; c < num_channels; c++) {
            ch_buf[c][s] = interleaved[s * num_channels + c];
        }
    }
}

/* --- Interleave: separate mono channels → 6ch interleaved --- */
static void interleave(int16_t *interleaved, const int16_t ch_buf[][FRAME_SAMPLES],
                       int num_channels, int frame_samples) {
    for (int s = 0; s < frame_samples; s++) {
        for (int c = 0; c < num_channels; c++) {
            interleaved[s * num_channels + c] = ch_buf[c][s];
        }
    }
}

/* --- Signal handler --- */
static void sig_handler(int sig) {
    (void)sig;
    g_running = 0;
}

/* --- Main --- */
int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    printf("aec_daemon: starting 6ch mode (rate=%d, frame=%d, filter=%d, mics=%d)\n",
           SAMPLE_RATE, FRAME_SAMPLES, FILTER_LENGTH, NUM_AEC_CHANNELS);

    ring_init(&g_ring);

    /* Initialize 4 independent speexdsp echo cancellers (one per mic channel) */
    SpeexEchoState *echo_st[NUM_AEC_CHANNELS];
    SpeexPreprocessState *pp_st[NUM_AEC_CHANNELS];
    int sr = SAMPLE_RATE;

    for (int i = 0; i < NUM_AEC_CHANNELS; i++) {
        echo_st[i] = speex_echo_state_init(FRAME_SAMPLES, FILTER_LENGTH);
        speex_echo_ctl(echo_st[i], SPEEX_ECHO_SET_SAMPLING_RATE, &sr);

        pp_st[i] = speex_preprocess_state_init(FRAME_SAMPLES, SAMPLE_RATE);
        speex_preprocess_ctl(pp_st[i], SPEEX_PREPROCESS_SET_ECHO_STATE, echo_st[i]);

        printf("  AEC instance %d for ch%d initialized\n", i, AEC_CH[i]);
    }

    /* Open mic capture (6ch from cap_dsnoop) and output (6ch to loopback) */
    snd_pcm_t *mic = open_alsa("cap_dsnoop", SND_PCM_STREAM_CAPTURE, MIC_CHANNELS);
    snd_pcm_t *out = open_alsa("hw:1,0,1", SND_PCM_STREAM_PLAYBACK, MIC_CHANNELS);

    if (!mic || !out) {
        fprintf(stderr, "main: failed to open mic/output devices\n");
        return 1;
    }

    /* Start reference thread (mono TTS capture) */
    pthread_t ref_tid;
    pthread_create(&ref_tid, NULL, ref_thread, NULL);

    /* Buffers */
    int16_t mic_interleaved[FRAME_SAMPLES * MIC_CHANNELS];
    int16_t out_interleaved[FRAME_SAMPLES * MIC_CHANNELS];
    int16_t ch_in[MIC_CHANNELS][FRAME_SAMPLES];
    int16_t ch_out[MIC_CHANNELS][FRAME_SAMPLES];
    int16_t ref_frame[FRAME_SAMPLES];
    int16_t aec_out[FRAME_SAMPLES];

    printf("aec_daemon: running (6ch)\n");

    while (g_running) {
        /* Read 6ch interleaved mic input */
        snd_pcm_sframes_t n = snd_pcm_readi(mic, mic_interleaved, FRAME_SAMPLES);
        if (n < 0) {
            if (alsa_recover(mic, n) < 0) break;
            continue;
        }
        if (n != FRAME_SAMPLES) continue;

        /* Deinterleave to per-channel buffers */
        deinterleave(mic_interleaved, ch_in, MIC_CHANNELS, FRAME_SAMPLES);

        /* Try to get TTS reference frame (mono) */
        int has_ref = ring_read(&g_ring, ref_frame);

        if (has_ref) {
            /* Apply AEC to each mic channel */
            for (int i = 0; i < NUM_AEC_CHANNELS; i++) {
                speex_echo_cancellation(echo_st[i], ch_in[AEC_CH[i]], ref_frame, aec_out);
                speex_preprocess_run(pp_st[i], aec_out);
                memcpy(ch_out[AEC_CH[i]], aec_out, FRAME_BYTES_MONO);
            }
        } else {
            /* No TTS playing — pass mic channels through directly */
            for (int i = 0; i < NUM_AEC_CHANNELS; i++) {
                memcpy(ch_out[AEC_CH[i]], ch_in[AEC_CH[i]], FRAME_BYTES_MONO);
            }
        }

        /* Passthrough channels (ch1: hw ref, ch5: unknown) */
        for (int i = 0; i < NUM_PASS_CHANNELS; i++) {
            memcpy(ch_out[PASS_CH[i]], ch_in[PASS_CH[i]], FRAME_BYTES_MONO);
        }

        /* Interleave and write 6ch output */
        interleave(out_interleaved, (const int16_t (*)[FRAME_SAMPLES])ch_out,
                   MIC_CHANNELS, FRAME_SAMPLES);

        snd_pcm_sframes_t w = snd_pcm_writei(out, out_interleaved, FRAME_SAMPLES);
        if (w < 0) {
            if (alsa_recover(out, w) < 0) break;
        }
    }

    printf("aec_daemon: shutting down\n");

    g_running = 0;
    pthread_join(ref_tid, NULL);

    snd_pcm_close(mic);
    snd_pcm_close(out);

    for (int i = 0; i < NUM_AEC_CHANNELS; i++) {
        speex_preprocess_state_destroy(pp_st[i]);
        speex_echo_state_destroy(echo_st[i]);
    }

    return 0;
}
