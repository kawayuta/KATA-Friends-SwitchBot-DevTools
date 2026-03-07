/*
 * aec_daemon.c — AEC (Acoustic Echo Cancellation) daemon using speexdsp
 *
 * Architecture:
 *   Reference thread: hw:1,1,0 (loopback cap, TTS ref) → ring buffer + dmix passthrough
 *   Main thread:      dsnoop (mic) + ring buffer → speex AEC → hw:1,0,1 (loopback out)
 *
 * Parameters: 16kHz, S16_LE, mono, 160 samples/frame (10ms), 2048 filter length (128ms)
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
#define CHANNELS      1
#define FRAME_SAMPLES 160        /* 10ms at 16kHz */
#define FILTER_LENGTH 2048       /* 128ms tail */
#define RING_FRAMES   256        /* ring buffer capacity in frames */
#define FRAME_BYTES   (FRAME_SAMPLES * sizeof(int16_t))

static volatile sig_atomic_t g_running = 1;

/* --- Ring buffer (single-producer, single-consumer) --- */
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
    memcpy(&r->buf[idx * FRAME_SAMPLES], frame, FRAME_BYTES);
    r->head++;
    /* if head overtakes tail, advance tail (drop oldest) */
    if (r->head - r->tail > RING_FRAMES)
        r->tail = r->head - RING_FRAMES;
    pthread_mutex_unlock(&r->mtx);
}

/* Returns 1 if a frame was read, 0 if empty */
static int ring_read(RingBuf *r, int16_t *frame) {
    pthread_mutex_lock(&r->mtx);
    if (r->tail == r->head) {
        pthread_mutex_unlock(&r->mtx);
        return 0;
    }
    unsigned int idx = r->tail % RING_FRAMES;
    memcpy(frame, &r->buf[idx * FRAME_SAMPLES], FRAME_BYTES);
    r->tail++;
    pthread_mutex_unlock(&r->mtx);
    return 1;
}

static RingBuf g_ring;

/* --- ALSA helpers --- */
static snd_pcm_t *open_alsa(const char *dev, snd_pcm_stream_t dir) {
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
    snd_pcm_hw_params_set_channels(pcm, params, CHANNELS);

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

/* --- Reference thread: capture TTS from loopback, feed ring buf + speaker --- */
static void *ref_thread(void *arg) {
    (void)arg;

    snd_pcm_t *cap = open_alsa("hw:1,1,0", SND_PCM_STREAM_CAPTURE);
    snd_pcm_t *play = open_alsa("softvol_ply", SND_PCM_STREAM_PLAYBACK);

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

        /* Store reference for AEC */
        ring_write(&g_ring, frame);

        /* Passthrough to speaker */
        snd_pcm_sframes_t w = snd_pcm_writei(play, frame, FRAME_SAMPLES);
        if (w < 0) {
            if (alsa_recover(play, w) < 0) break;
        }
    }

    snd_pcm_close(cap);
    snd_pcm_close(play);
    return NULL;
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

    printf("aec_daemon: starting (rate=%d, frame=%d, filter=%d)\n",
           SAMPLE_RATE, FRAME_SAMPLES, FILTER_LENGTH);

    ring_init(&g_ring);

    /* Initialize speexdsp echo canceller */
    SpeexEchoState *echo_st = speex_echo_state_init(FRAME_SAMPLES, FILTER_LENGTH);
    int sr = SAMPLE_RATE;
    speex_echo_ctl(echo_st, SPEEX_ECHO_SET_SAMPLING_RATE, &sr);

    /* Optional: preprocessor for residual echo suppression */
    SpeexPreprocessState *pp_st = speex_preprocess_state_init(FRAME_SAMPLES, SAMPLE_RATE);
    speex_preprocess_ctl(pp_st, SPEEX_PREPROCESS_SET_ECHO_STATE, echo_st);

    /* Open mic capture (dsnoop) and cleaned audio output (loopback) */
    snd_pcm_t *mic = open_alsa("cap_dsnoop", SND_PCM_STREAM_CAPTURE);
    snd_pcm_t *out = open_alsa("hw:1,0,1", SND_PCM_STREAM_PLAYBACK);

    if (!mic || !out) {
        fprintf(stderr, "main: failed to open mic/output devices\n");
        return 1;
    }

    /* Start reference thread */
    pthread_t ref_tid;
    pthread_create(&ref_tid, NULL, ref_thread, NULL);

    int16_t mic_frame[FRAME_SAMPLES];
    int16_t ref_frame[FRAME_SAMPLES];
    int16_t out_frame[FRAME_SAMPLES];

    printf("aec_daemon: running\n");

    while (g_running) {
        /* Read mic input */
        snd_pcm_sframes_t n = snd_pcm_readi(mic, mic_frame, FRAME_SAMPLES);
        if (n < 0) {
            if (alsa_recover(mic, n) < 0) break;
            continue;
        }
        if (n != FRAME_SAMPLES) continue;

        /* Try to get reference frame */
        if (ring_read(&g_ring, ref_frame)) {
            /* AEC processing */
            speex_echo_cancellation(echo_st, mic_frame, ref_frame, out_frame);
            speex_preprocess_run(pp_st, out_frame);
        } else {
            /* No TTS playing — pass mic through directly */
            memcpy(out_frame, mic_frame, FRAME_BYTES);
        }

        /* Write cleaned audio to loopback for pet_voice */
        snd_pcm_sframes_t w = snd_pcm_writei(out, out_frame, FRAME_SAMPLES);
        if (w < 0) {
            if (alsa_recover(out, w) < 0) break;
        }
    }

    printf("aec_daemon: shutting down\n");

    g_running = 0;
    pthread_join(ref_tid, NULL);

    snd_pcm_close(mic);
    snd_pcm_close(out);
    speex_preprocess_state_destroy(pp_st);
    speex_echo_state_destroy(echo_st);

    return 0;
}
