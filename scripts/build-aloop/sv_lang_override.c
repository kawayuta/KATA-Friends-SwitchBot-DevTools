#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <syslog.h>

// sherpa-onnx v1.12.15 struct definitions (fields before sense_voice only)
typedef struct { int32_t sample_rate; int32_t feature_dim; } FeatureConfig;
typedef struct { const char *encoder; const char *decoder; const char *joiner; } TransducerConfig;
typedef struct { const char *model; } ParaformerConfig;
typedef struct { const char *model; } NemoCtcConfig;
typedef struct { const char *encoder; const char *decoder; const char *language; const char *task; int32_t tail_paddings; } WhisperConfig;
typedef struct { const char *model; } TdnnConfig;
typedef struct { const char *model; const char *language; int32_t use_itn; } SenseVoiceConfig;

typedef struct {
    TransducerConfig transducer;
    ParaformerConfig paraformer;
    NemoCtcConfig nemo_ctc;
    WhisperConfig whisper;
    TdnnConfig tdnn;
    const char *tokens;
    int32_t num_threads;
    int32_t debug;
    const char *provider;
    const char *model_type;
    const char *modeling_unit;
    const char *bpe_vocab;
    const char *telespeech_ctc;
    SenseVoiceConfig sense_voice;
    // ... remaining fields omitted
} ModelConfig;

typedef struct {
    FeatureConfig feat_config;
    ModelConfig model_config;
    // ... remaining fields omitted
} RecognizerConfig;

typedef void* (*create_fn)(const RecognizerConfig*);

static char lang_buf[32] = "auto";

void* SherpaOnnxCreateOfflineRecognizer(const RecognizerConfig *config) {
    create_fn real_fn = (create_fn)dlsym(RTLD_NEXT, "SherpaOnnxCreateOfflineRecognizer");

    // Read language from config file
    FILE *f = fopen("/data/devtools/asr_language.conf", "r");
    if (f) {
        if (fgets(lang_buf, sizeof(lang_buf), f)) {
            // Strip newline
            char *nl = strchr(lang_buf, '\n');
            if (nl) *nl = '\0';
        }
        fclose(f);
    }

    // Override language (cast away const for the override)
    RecognizerConfig *cfg = (RecognizerConfig*)config;
    if (lang_buf[0] && strcmp(lang_buf, "auto") != 0) {
        cfg->model_config.sense_voice.language = lang_buf;
        syslog(LOG_INFO, "[sv_lang] Override SenseVoice language: %s", lang_buf);
    } else {
        syslog(LOG_INFO, "[sv_lang] SenseVoice language: %s (no override)", lang_buf);
    }

    return real_fn(config);
}
