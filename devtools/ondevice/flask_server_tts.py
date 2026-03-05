#!/bin/env python3
"""
Kata Friends TTS/ASR Service — Qwen3-TTS 0.6B (RKNN) + Whisper (RKNN)

Runs on device at port 8084. Uses NPU via librknnrt.so (ctypes).
Models are lazy-loaded on first request.

Pipeline based on sivasub987/Qwen3-TTS-0.6B-ONNX-INT8 reference scripts.
Config: hidden_size=1024, 28 talker layers, 16 code groups, 24kHz output.

Endpoints:
  POST /tts/synthesize   — Text → speech synthesis (NPU), optional aplay
  GET  /tts/status       — Model load state
  POST /tts/unload       — Unload models to free NPU memory
  POST /asr/transcribe   — Whisper speech-to-text (NPU)
"""

import ctypes
import sys
import os
import subprocess
import threading
import time
import argparse
import wave
import json
import base64
import tempfile
import logging
from logging.handlers import TimedRotatingFileHandler

import numpy as np
from flask import Flask, request, jsonify

os.environ["TZ"] = "Asia/Shanghai"
time.tzset()

app = Flask(__name__)

# --- Logging ---
logger = logging.getLogger("tts")
logger.setLevel(logging.INFO)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_sh)

LOG_DIR = "/data/cache/log"
if os.path.isdir(LOG_DIR):
    _fh = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "tts_server.log"), when="midnight", backupCount=7
    )
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

# --- Config ---
TTS_MODEL_DIR = "/data/ai_brain/tts"
ASR_MODEL_DIR = "/data/ai_brain/asr"
REFERENCE_DIR = "/data/devtools/tts"
REFERENCE_WAV = os.path.join(REFERENCE_DIR, "reference.wav")
REFERENCE_TEXT = os.path.join(REFERENCE_DIR, "reference_text.txt")

RKNN_LIB_PATH = "/app/opt/wlab/sweepbot/lib/librknnrt.so"
SAMPLE_RATE_TTS = 24000  # Qwen3-TTS output
SAMPLE_RATE_ASR = 16000  # Whisper input

# Model constants from config.json
HIDDEN_SIZE = 1024
NUM_CODE_GROUPS = 16
TALKER_VOCAB_SIZE = 3072
CODE_PREDICTOR_VOCAB_SIZE = 2048
TALKER_NUM_LAYERS = 28
TALKER_NUM_KV_HEADS = 8
TALKER_HEAD_DIM = 128

# Special token IDs from config.json
CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150
CODEC_PAD_ID = 2148

# --- RKNN ctypes definitions (same as flask_server_diary.py) ---
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_INT16 = 4
RKNN_TENSOR_INT64 = 7
RKNN_TENSOR_NHWC = 1
RKNN_TENSOR_NCHW = 0
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_NPU_CORE_0_1 = 3

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256


class RKNNInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RKNNTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RKNNInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


# --- RKNN Model Loader ---
_rknn_lib = None
_lock = threading.Lock()
_tts_models = {}  # name -> model_dict
_asr_models = {}  # name -> model_dict
_tts_loaded = False
_asr_loaded = False

# Use onnxruntime as fallback for models that fail RKNN conversion
_ort_sessions = {}
try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# Tokenizer
_tokenizer = None


def _get_rknn_lib():
    """Load librknnrt.so once."""
    global _rknn_lib
    if _rknn_lib is not None:
        return _rknn_lib
    try:
        _rknn_lib = ctypes.CDLL(RKNN_LIB_PATH)
        logger.info(f"Loaded {RKNN_LIB_PATH}")
    except OSError as e:
        logger.error(f"Cannot load librknnrt.so: {e}")
        return None
    return _rknn_lib


def _load_rknn_model(model_path, use_dual_core=False):
    """Load a single .rknn model, return context dict."""
    lib = _get_rknn_lib()
    if lib is None:
        return None

    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}")
        return None

    ctx = ctypes.c_uint64(0)
    ret = lib.rknn_init(
        ctypes.byref(ctx),
        model_path.encode(),
        ctypes.c_uint32(0),
        ctypes.c_uint32(0),
        ctypes.c_void_p(None),
    )
    if ret != 0:
        logger.error(f"rknn_init failed for {model_path}: {ret}")
        return None

    if use_dual_core:
        lib.rknn_set_core_mask(ctx, ctypes.c_int(RKNN_NPU_CORE_0_1))

    # Query IO
    io_num = RKNNInputOutputNum()
    lib.rknn_query(
        ctx,
        ctypes.c_int(RKNN_QUERY_IN_OUT_NUM),
        ctypes.byref(io_num),
        ctypes.c_uint32(ctypes.sizeof(io_num)),
    )

    # Query input attrs
    in_attrs = []
    for i in range(io_num.n_input):
        attr = RKNNTensorAttr()
        attr.index = i
        lib.rknn_query(
            ctx,
            ctypes.c_int(RKNN_QUERY_INPUT_ATTR),
            ctypes.byref(attr),
            ctypes.c_uint32(ctypes.sizeof(attr)),
        )
        in_attrs.append(attr)

    # Query output attrs
    out_attrs = []
    for i in range(io_num.n_output):
        attr = RKNNTensorAttr()
        attr.index = i
        lib.rknn_query(
            ctx,
            ctypes.c_int(RKNN_QUERY_OUTPUT_ATTR),
            ctypes.byref(attr),
            ctypes.c_uint32(ctypes.sizeof(attr)),
        )
        out_attrs.append(attr)

    name = os.path.basename(model_path)
    logger.info(
        f"Loaded {name}: {io_num.n_input} inputs, {io_num.n_output} outputs, ctx={ctx.value}"
    )

    return {
        "lib": lib,
        "ctx": ctx,
        "n_input": io_num.n_input,
        "n_output": io_num.n_output,
        "in_attrs": in_attrs,
        "out_attrs": out_attrs,
        "path": model_path,
    }


def _run_rknn(model_dict, inputs_list):
    """
    Run RKNN inference.
    inputs_list: list of (numpy_array, rknn_type) tuples.
    Returns list of numpy arrays (float32 outputs).
    """
    lib = model_dict["lib"]
    ctx = model_dict["ctx"]
    n_in = model_dict["n_input"]
    n_out = model_dict["n_output"]

    if len(inputs_list) != n_in:
        logger.error(f"Expected {n_in} inputs, got {len(inputs_list)}")
        return None

    # Set inputs
    InputArray = RKNNInput * n_in
    inputs = InputArray()
    bufs = []  # keep references alive
    for i, (arr, tensor_type) in enumerate(inputs_list):
        arr_c = np.ascontiguousarray(arr)
        bufs.append(arr_c)
        inputs[i].index = i
        inputs[i].type = tensor_type
        inputs[i].fmt = RKNN_TENSOR_NCHW
        inputs[i].size = arr_c.nbytes
        inputs[i].buf = arr_c.ctypes.data
        inputs[i].pass_through = 0

    ret = lib.rknn_inputs_set(ctx, ctypes.c_uint32(n_in), inputs)
    if ret < 0:
        logger.error(f"rknn_inputs_set failed: {ret}")
        return None

    # Run
    ret = lib.rknn_run(ctx, ctypes.c_void_p(None))
    if ret < 0:
        logger.error(f"rknn_run failed: {ret}")
        return None

    # Get outputs
    OutputArray = RKNNOutput * n_out
    outputs = OutputArray()
    for j in range(n_out):
        outputs[j].want_float = 1
        outputs[j].is_prealloc = 0

    ret = lib.rknn_outputs_get(
        ctx, ctypes.c_uint32(n_out), outputs, ctypes.c_void_p(None)
    )
    if ret < 0:
        logger.error(f"rknn_outputs_get failed: {ret}")
        return None

    # Copy output data
    result = []
    for j in range(n_out):
        out_attr = model_dict["out_attrs"][j]
        dims = [out_attr.dims[d] for d in range(out_attr.n_dims)]
        n_floats = 1
        for d in dims:
            n_floats *= d
        arr = np.zeros(n_floats, dtype=np.float32)
        ctypes.memmove(arr.ctypes.data, outputs[j].buf, n_floats * 4)
        arr = arr.reshape(dims)
        result.append(arr)

    lib.rknn_outputs_release(ctx, ctypes.c_uint32(n_out), outputs)
    return result


def _unload_rknn(model_dict):
    """Release a RKNN model context."""
    if model_dict and "lib" in model_dict and "ctx" in model_dict:
        model_dict["lib"].rknn_destroy(model_dict["ctx"])


# --- TTS Model Management ---
TTS_MODEL_FILES = {
    "text_project": "text_project_q",
    "tokenizer12hz_encode": "tokenizer12hz_encode_q",
    "tokenizer12hz_decode": "tokenizer12hz_decode_q",
    "speaker_encoder": "speaker_encoder_q",
    "codec_embed": "codec_embed_q",
    "code_predictor_embed": "code_predictor_embed_q",
    "code_predictor": "code_predictor_q",
    "talker_prefill": "talker_prefill_q",
    "talker_decode": "talker_decode_q",
}

# Models that benefit from dual NPU cores
DUAL_CORE_MODELS = {"talker_prefill", "talker_decode", "tokenizer12hz_decode"}


def _load_tts_models():
    """Load all TTS RKNN models (lazy, called on first request)."""
    global _tts_loaded, _tts_models, _tokenizer
    if _tts_loaded:
        return True

    logger.info("Loading TTS models...")
    missing = []

    for name, basename in TTS_MODEL_FILES.items():
        rknn_path = os.path.join(TTS_MODEL_DIR, f"{basename}.rknn")
        onnx_path = os.path.join(TTS_MODEL_DIR, f"{basename}.onnx")
        dual = name in DUAL_CORE_MODELS

        if os.path.exists(rknn_path):
            model = _load_rknn_model(rknn_path, use_dual_core=dual)
            if model is not None:
                _tts_models[name] = model
                continue

        # ONNX fallback
        if HAS_ORT and os.path.exists(onnx_path):
            logger.info(f"  {name}: using ONNX fallback (CPU)")
            _ort_sessions[name] = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            continue

        missing.append(name)

    if len(missing) == len(TTS_MODEL_FILES):
        logger.error("No TTS models loaded at all!")
        return False

    if missing:
        logger.warning(f"Missing TTS models (will skip): {missing}")

    # Load tokenizer
    _load_tokenizer()

    _tts_loaded = True
    logger.info(
        f"TTS models loaded: RKNN={list(_tts_models.keys())}, ORT={list(_ort_sessions.keys())}"
    )
    return True


def _load_tokenizer():
    """Load HuggingFace tokenizer for Qwen3-TTS."""
    global _tokenizer
    try:
        from transformers import AutoTokenizer

        # Try loading from model directory (vocab.json + merges.txt + tokenizer_config.json)
        tokenizer_dir = TTS_MODEL_DIR
        if os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            _tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer from {tokenizer_dir}")
            return
    except Exception as e:
        logger.warning(f"Could not load HF tokenizer: {e}")

    # Fallback: try loading vocab.json manually for basic tokenization
    vocab_path = os.path.join(TTS_MODEL_DIR, "vocab.json")
    if os.path.exists(vocab_path):
        logger.info("Using manual vocab.json tokenizer (basic)")
    else:
        logger.warning("No tokenizer available. Will use byte-level encoding.")


def _tokenize_text(text):
    """Tokenize text to int64 array."""
    if _tokenizer is not None:
        ids = _tokenizer.encode(text, add_special_tokens=False)
        return np.array([ids], dtype=np.int64)
    # Byte-level fallback
    text_bytes = text.encode("utf-8")
    ids = [b % 1000 for b in text_bytes]
    return np.array([ids[:512]], dtype=np.int64)


def _unload_tts_models():
    """Unload all TTS models to free NPU memory."""
    global _tts_loaded, _tts_models, _ort_sessions
    for name, m in _tts_models.items():
        _unload_rknn(m)
        logger.info(f"  Unloaded TTS RKNN: {name}")
    _tts_models.clear()
    _ort_sessions.clear()
    _tts_loaded = False
    logger.info("All TTS models unloaded.")


# --- Model runner (RKNN or ORT) ---


def _run_model(name, inputs_dict):
    """
    Run model by name. inputs_dict maps input_name → numpy array.
    For RKNN: inputs are passed in order with appropriate tensor types.
    For ORT: inputs are passed by name.
    Returns list of numpy arrays.
    """
    if name in _tts_models:
        # RKNN: map numpy dtypes to RKNN tensor types
        dtype_map = {
            np.float32: RKNN_TENSOR_FLOAT32,
            np.float16: RKNN_TENSOR_FLOAT16,
            np.int64: RKNN_TENSOR_INT64,
            np.int32: RKNN_TENSOR_INT64,  # treat int32 as int64 for RKNN
        }
        inputs_list = []
        for arr in inputs_dict.values():
            tensor_type = dtype_map.get(arr.dtype.type, RKNN_TENSOR_FLOAT32)
            inputs_list.append((arr, tensor_type))
        return _run_rknn(_tts_models[name], inputs_list)

    elif name in _ort_sessions:
        sess = _ort_sessions[name]
        return sess.run(None, inputs_dict)

    else:
        logger.error(f"Model '{name}' not loaded")
        return None


# --- TTS Inference Pipeline ---


def _load_reference_audio_24k(wav_path):
    """Load reference WAV file, return numpy float32 array at 24kHz."""
    if not os.path.exists(wav_path):
        return None
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-ar", str(SAMPLE_RATE_TTS), "-ac", "1",
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if proc.returncode != 0:
        logger.error(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _compute_mel_spectrogram(audio, sr=24000, n_fft=1024, hop_length=256, n_mels=128):
    """Compute mel spectrogram for speaker encoder (128 mel bins)."""
    window = np.hanning(n_fft)
    pad_len = n_fft // 2
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="reflect")

    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    magnitudes = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)

    for i in range(n_frames):
        frame = audio_padded[i * hop_length : i * hop_length + n_fft] * window
        spectrum = np.fft.rfft(frame)
        magnitudes[:, i] = np.abs(spectrum) ** 2

    mel_filters = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = mel_filters @ magnitudes

    # Log mel (power_to_db style)
    mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    mel_db -= mel_db.max()  # Normalize to max=0
    return mel_db.astype(np.float32)


def _mel_filterbank(sr, n_fft, n_mels):
    """Create mel filterbank matrix."""

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            if bin_points[i + 1] != bin_points[i]:
                filters[i, j] = (j - bin_points[i]) / (
                    bin_points[i + 1] - bin_points[i]
                )
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            if bin_points[i + 2] != bin_points[i + 1]:
                filters[i, j] = (bin_points[i + 2] - j) / (
                    bin_points[i + 2] - bin_points[i + 1]
                )
    return filters


def synthesize_speech(text, ref_wav_path=None, ref_text=None, max_steps=2048):
    """
    Full TTS pipeline: text → codec tokens → PCM audio.

    Pipeline (from sample_inference.py / full_tts_test.py):
      1. Tokenize text → input_ids
      2. text_project(input_ids) → text_embeds [1, seq, 1024]
      3. Load reference audio → 24kHz float32
      4. speaker_encoder(mels) → spk_emb [1, 1024]
      5. tokenizer12hz_encode(audio, mask) → ref_codes [1, n_codebooks, n_frames]
      6. talker_prefill(embeds, mask) → logits + KV cache
      7. talker_decode loop → generate codec token sequence
      8. code_predictor → predict sub-codebook codes for each frame
      9. tokenizer12hz_decode(codes) → 24kHz PCM

    Returns (pcm_float32_24khz, sample_rate) or (None, None) on error.
    """
    if not _tts_loaded:
        if not _load_tts_models():
            return None, None

    t0 = time.time()

    if ref_wav_path is None:
        ref_wav_path = REFERENCE_WAV
    if ref_text is None and os.path.exists(REFERENCE_TEXT):
        with open(REFERENCE_TEXT, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

    # 1. Tokenize text
    input_ids = _tokenize_text(text)
    logger.info(f"Text tokens: {input_ids.shape}")

    # 2. Text projection
    text_embeds = _run_model("text_project", {"input_ids": input_ids})
    if text_embeds is None:
        return None, None
    text_embeds = text_embeds[0]  # [1, seq, 1024]
    logger.info(f"Text embeds: {text_embeds.shape}")

    # 3. Load reference audio
    ref_audio = _load_reference_audio_24k(ref_wav_path)
    if ref_audio is None:
        logger.error("No reference audio available")
        return None, None

    # 4. Speaker encoder
    mel = _compute_mel_spectrogram(ref_audio, sr=SAMPLE_RATE_TTS, n_mels=128)
    # Pad/truncate to [1, 128, 128]
    target_frames = 128
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
    else:
        # Take center portion
        start = (mel.shape[1] - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    mel_input = mel[np.newaxis, :, :]  # [1, 128, 128]

    spk_out = _run_model("speaker_encoder", {"mels": mel_input.astype(np.float32)})
    if spk_out is None:
        return None, None
    spk_emb = spk_out[0]  # [1, 1024]
    logger.info(f"Speaker embedding: {spk_emb.shape}")

    # 5. Encode reference audio → codec tokens
    # Pad/truncate to 1s (24000 samples)
    ref_padded = np.zeros(SAMPLE_RATE_TTS, dtype=np.float32)
    ref_len = min(len(ref_audio), SAMPLE_RATE_TTS)
    ref_padded[:ref_len] = ref_audio[:ref_len]
    audio_input = ref_padded.reshape(1, -1)  # [1, 24000]
    padding_mask = np.ones_like(audio_input, dtype=np.int64)
    padding_mask[0, ref_len:] = 0  # Mask padding

    codec_out = _run_model(
        "tokenizer12hz_encode",
        {"input_values": audio_input, "padding_mask": padding_mask},
    )
    if codec_out is None:
        return None, None
    ref_codes = codec_out[0]  # [1, n_codebooks, n_frames]
    logger.info(f"Reference codec: {ref_codes.shape}")

    # 6. Build prefill sequence
    # Combine text embeddings with codec conditioning
    # Get codec embedding for first frame
    first_code = np.array([[int(ref_codes[0, 0, 0])]], dtype=np.int64)  # [1, 1]
    codec_emb_out = _run_model("codec_embed", {"input_ids": first_code})
    if codec_emb_out is None:
        return None, None
    codec_emb = codec_emb_out[0]  # [1, 1, D] or [1, D]
    if codec_emb.ndim == 2:
        codec_emb = codec_emb[:, np.newaxis, :]
    logger.info(f"Codec embed: {codec_emb.shape}")

    # Concatenate: text_embeds + speaker(broadcast) + codec_emb
    if spk_emb.ndim == 2:
        spk_emb = spk_emb[:, np.newaxis, :]  # [1, 1, 1024]
    prefill_seq = np.concatenate([text_embeds, spk_emb, codec_emb], axis=1)  # [1, N, 1024]

    seq_len = prefill_seq.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    logger.info(f"Prefill seq: {prefill_seq.shape}")

    # 7. Talker prefill
    prefill_out = _run_model(
        "talker_prefill",
        {
            "inputs_embeds": prefill_seq.astype(np.float32),
            "attention_mask": attention_mask,
        },
    )
    if prefill_out is None:
        return None, None

    logits = prefill_out[0]  # [1, seq, vocab_size]
    kv_cache = prefill_out[1:] if len(prefill_out) > 1 else []
    logger.info(
        f"Prefill done: logits={logits.shape}, kv_cache={len(kv_cache)} tensors"
    )

    # 8. Autoregressive decode loop
    generated_codes = []

    # Get first token from prefill logits
    last_logits = logits[0, -1, :]  # [vocab_size]
    token_id = int(np.argmax(last_logits))
    if token_id == CODEC_EOS_ID or token_id == 0:
        logger.warning("Prefill produced EOS immediately")
    else:
        generated_codes.append(token_id)

    for step in range(max_steps):
        # Embed current token
        cur_code = np.array([[token_id]], dtype=np.int64)
        emb_out = _run_model("codec_embed", {"input_ids": cur_code})
        if emb_out is None:
            break
        cur_emb = emb_out[0]
        if cur_emb.ndim == 2:
            cur_emb = cur_emb[:, np.newaxis, :]  # [1, 1, 1024]

        # Build decode inputs
        decode_mask = np.ones((1, seq_len + step + 1), dtype=np.int64)
        decode_inputs = {
            "inputs_embeds": cur_emb.astype(np.float32),
            "attention_mask": decode_mask,
        }
        # Add KV cache inputs
        for i, kv in enumerate(kv_cache):
            decode_inputs[f"past_key_values.{i}"] = kv

        decode_out = _run_model("talker_decode", decode_inputs)
        if decode_out is None:
            break

        step_logits = decode_out[0]
        kv_cache = decode_out[1:] if len(decode_out) > 1 else kv_cache

        # Greedy decode
        if step_logits.ndim >= 2:
            token_id = int(np.argmax(step_logits[0, -1, :]))
        else:
            token_id = int(np.argmax(step_logits.reshape(-1)))

        if token_id == CODEC_EOS_ID or token_id == 0:
            break

        generated_codes.append(token_id)

    logger.info(
        f"Generated {len(generated_codes)} primary codes in {time.time()-t0:.1f}s"
    )

    if not generated_codes:
        logger.warning("No tokens generated")
        return None, None

    # 9. Sub-codebook prediction
    n_frames = len(generated_codes)
    full_codes = np.zeros((1, NUM_CODE_GROUPS, n_frames), dtype=np.int64)
    full_codes[0, 0, :] = generated_codes

    for frame_idx in range(n_frames):
        for cb in range(1, NUM_CODE_GROUPS):
            # Get embedding for previous code
            prev_code = np.array([[int(full_codes[0, cb - 1, frame_idx])]], dtype=np.int64)
            gen_step = np.array([cb - 1], dtype=np.int64)

            emb_out = _run_model(
                "code_predictor_embed",
                {"input_ids": prev_code, "generation_step": gen_step},
            )
            if emb_out is None:
                break

            hidden = emb_out[0]
            if hidden.ndim == 2:
                hidden = hidden[:, np.newaxis, :]  # [1, 1, 1024]

            pred_out = _run_model(
                "code_predictor",
                {"inputs_embeds": hidden.astype(np.float32), "generation_step": gen_step},
            )
            if pred_out is None:
                break

            pred_token = int(np.argmax(pred_out[0].reshape(-1)))
            full_codes[0, cb, frame_idx] = pred_token

    logger.info(f"Sub-codebook prediction done for {n_frames} frames")

    # 10. Vocoder: codec tokens → PCM
    vocoder_out = _run_model(
        "tokenizer12hz_decode",
        {"audio_codes": full_codes.astype(np.int64)},
    )
    if vocoder_out is None:
        return None, None

    pcm = vocoder_out[0].flatten().astype(np.float32)

    elapsed = time.time() - t0
    duration = len(pcm) / SAMPLE_RATE_TTS
    rtf = elapsed / max(duration, 0.01)
    logger.info(f"TTS complete: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})")

    return pcm, SAMPLE_RATE_TTS


def _save_wav(pcm_float32, sample_rate, path):
    """Save float32 PCM to WAV file."""
    pcm_clipped = np.clip(pcm_float32, -1.0, 1.0)
    pcm_int16 = (pcm_clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())


# --- ASR (Whisper) ---
WHISPER_N_FFT = 400
WHISPER_HOP = 160
WHISPER_N_MELS = 80
WHISPER_SR = 16000
WHISPER_MAX_AUDIO = 30  # seconds


def _load_asr_models():
    """Load Whisper RKNN models."""
    global _asr_loaded, _asr_models

    if _asr_loaded:
        return True

    logger.info("Loading ASR (Whisper) models...")
    for fname in ["whisper_encoder.rknn", "whisper_decoder.rknn"]:
        path = os.path.join(ASR_MODEL_DIR, fname)
        name = fname.replace(".rknn", "")
        model = _load_rknn_model(path)
        if model is None:
            logger.error(f"Failed to load ASR model: {fname}")
            return False
        _asr_models[name] = model

    _asr_loaded = True
    logger.info(f"ASR models loaded: {list(_asr_models.keys())}")
    return True


def _whisper_mel(audio_16k):
    """Compute Whisper-compatible log-mel spectrogram."""
    max_samples = WHISPER_SR * WHISPER_MAX_AUDIO
    if len(audio_16k) > max_samples:
        audio_16k = audio_16k[:max_samples]

    audio_padded = np.zeros(max_samples, dtype=np.float32)
    audio_padded[: len(audio_16k)] = audio_16k

    window = np.hanning(WHISPER_N_FFT)
    n_frames = 1 + (len(audio_padded) - WHISPER_N_FFT) // WHISPER_HOP
    magnitudes = np.zeros((WHISPER_N_FFT // 2 + 1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * WHISPER_HOP
        frame = audio_padded[start : start + WHISPER_N_FFT] * window
        spectrum = np.fft.rfft(frame)
        magnitudes[:, i] = np.abs(spectrum) ** 2

    mel_filters = _mel_filterbank(WHISPER_SR, WHISPER_N_FFT, WHISPER_N_MELS)
    mel_spec = mel_filters @ magnitudes

    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.astype(np.float32)


_whisper_vocab = None


def _load_whisper_vocab():
    """Load Whisper vocabulary."""
    global _whisper_vocab
    vocab_path = os.path.join(ASR_MODEL_DIR, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            _whisper_vocab = json.load(f)
        logger.info(f"Loaded Whisper vocab: {len(_whisper_vocab)} tokens")
    else:
        logger.warning("Whisper vocab.json not found")
        _whisper_vocab = {}


def _decode_whisper_tokens(token_ids):
    """Decode Whisper token IDs to text."""
    if _whisper_vocab is None:
        _load_whisper_vocab()

    if not _whisper_vocab:
        return "".join(chr(t) for t in token_ids if 0 <= t < 128)

    pieces = []
    for tid in token_ids:
        token = _whisper_vocab.get(str(tid), "")
        if token.startswith("<|") and token.endswith("|>"):
            continue
        pieces.append(token)

    text = "".join(pieces).replace("\u0120", " ").strip()
    return text


def transcribe_audio(wav_path):
    """Transcribe audio using Whisper RKNN."""
    if not _asr_loaded:
        if not _load_asr_models():
            return None

    t0 = time.time()

    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-ar", str(WHISPER_SR), "-ac", "1",
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if proc.returncode != 0:
        logger.error(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
        return None

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    logger.info(f"ASR input: {len(audio)} samples ({len(audio)/WHISPER_SR:.1f}s)")

    mel = _whisper_mel(audio)
    mel_input = mel[np.newaxis, :, :]  # [1, 80, 3000]

    encoder = _asr_models.get("whisper_encoder")
    if encoder is None:
        return None

    enc_out = _run_rknn(encoder, [(mel_input, RKNN_TENSOR_FLOAT32)])
    if enc_out is None:
        return None
    encoder_output = enc_out[0]
    logger.info(f"Whisper encoder: {encoder_output.shape}")

    decoder = _asr_models.get("whisper_decoder")
    if decoder is None:
        return None

    SOT_TOKEN = 50258
    NO_TIMESTAMPS = 50363
    EOT_TOKEN = 50257

    token_ids = [SOT_TOKEN, NO_TIMESTAMPS]

    for step in range(224):
        tokens_arr = np.array([token_ids], dtype=np.int64)
        dec_out = _run_rknn(
            decoder,
            [
                (encoder_output, RKNN_TENSOR_FLOAT32),
                (tokens_arr, RKNN_TENSOR_INT64),
            ],
        )
        if dec_out is None:
            break

        logits = dec_out[0]
        next_logits = logits[0, -1, :] if logits.ndim == 3 else logits.reshape(-1)
        next_token = int(np.argmax(next_logits))

        if next_token == EOT_TOKEN:
            break
        token_ids.append(next_token)

    output_tokens = token_ids[2:]
    text = _decode_whisper_tokens(output_tokens)

    elapsed = time.time() - t0
    logger.info(f"ASR done: '{text[:80]}' in {elapsed:.1f}s")
    return text


# --- Flask Endpoints ---
is_busy = False


@app.route("/tts/synthesize", methods=["POST"])
def tts_synthesize():
    """Synthesize speech from text."""
    global is_busy
    if is_busy:
        return jsonify({"error": "TTS server busy"}), 503

    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    play_device = data.get("play_device", False)
    ref_text = data.get("ref_text")

    _lock.acquire()
    is_busy = True
    try:
        pcm, sr = synthesize_speech(text, ref_text=ref_text)
        if pcm is None:
            return jsonify({"error": "TTS synthesis failed"}), 500

        wav_path = tempfile.mktemp(suffix=".wav", prefix="tts_")
        _save_wav(pcm, sr, wav_path)

        duration = len(pcm) / sr

        if play_device:
            subprocess.Popen(["aplay", wav_path])

        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        if not play_device:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        return jsonify(
            {
                "status": "ok",
                "audio_base64": audio_b64,
                "duration_sec": round(duration, 2),
                "sample_rate": sr,
            }
        )

    except Exception as e:
        logger.exception("TTS synthesis error")
        return jsonify({"error": str(e)}), 500
    finally:
        is_busy = False
        _lock.release()


@app.route("/tts/status", methods=["GET"])
def tts_status():
    """Check TTS model status."""
    return jsonify(
        {
            "tts_loaded": _tts_loaded,
            "tts_models": list(_tts_models.keys()),
            "tts_ort_fallback": list(_ort_sessions.keys()),
            "asr_loaded": _asr_loaded,
            "asr_models": list(_asr_models.keys()),
            "has_reference": os.path.exists(REFERENCE_WAV),
            "busy": is_busy,
        }
    )


@app.route("/tts/unload", methods=["POST"])
def tts_unload():
    """Unload TTS models to free NPU memory."""
    _lock.acquire()
    try:
        _unload_tts_models()
        return jsonify({"status": "ok", "message": "TTS models unloaded"})
    finally:
        _lock.release()


@app.route("/asr/transcribe", methods=["POST"])
def asr_transcribe():
    """Transcribe audio file using Whisper."""
    global is_busy
    if is_busy:
        return jsonify({"error": "Server busy"}), 503

    data = request.get_json(force=True)
    wav_path = data.get("wav_path", "")

    if not wav_path or not os.path.exists(wav_path):
        return jsonify({"error": f"WAV file not found: {wav_path}"}), 400

    _lock.acquire()
    is_busy = True
    try:
        text = transcribe_audio(wav_path)
        if text is None:
            return jsonify({"error": "ASR transcription failed"}), 500
        return jsonify({"text": text})
    except Exception as e:
        logger.exception("ASR transcription error")
        return jsonify({"error": str(e)}), 500
    finally:
        is_busy = False
        _lock.release()


@app.route("/", methods=["GET"])
def index():
    """Health check."""
    return jsonify({"service": "tts_asr", "status": "ok"})


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kata Friends TTS/ASR Server")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tts-dir", default=TTS_MODEL_DIR, help="TTS model directory")
    parser.add_argument("--asr-dir", default=ASR_MODEL_DIR, help="ASR model directory")
    parser.add_argument("--preload", action="store_true", help="Preload models at startup")
    args = parser.parse_args()

    TTS_MODEL_DIR = args.tts_dir
    ASR_MODEL_DIR = args.asr_dir

    if args.preload:
        logger.info("Preloading models...")
        _load_tts_models()
        _load_asr_models()

    logger.info(f"Starting TTS/ASR server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
