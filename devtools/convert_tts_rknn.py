#!/usr/bin/env python3
"""
Qwen3-TTS 0.6B ONNX INT8 → RKNN conversion script.

Run on x86 Linux host with rknn-toolkit2 installed.
Source: HuggingFace sivasub987/Qwen3-TTS-0.6B-ONNX-INT8

Input names/shapes determined from the reference inference scripts
(sample_inference.py, full_tts_test.py) in the HuggingFace repo.

Usage:
    pip install rknn-toolkit2 huggingface_hub onnx
    python convert_tts_rknn.py [--output-dir ./tts_rknn_models]

    # Convert a single model:
    python convert_tts_rknn.py --model speaker_encoder

    # Use already-downloaded ONNX files:
    python convert_tts_rknn.py --onnx-dir ./onnx_models

    # Auto-detect input shapes from ONNX (recommended):
    python convert_tts_rknn.py --auto-shape
"""

import argparse
import os
import sys

try:
    from rknn.api import RKNN
except ImportError:
    print("ERROR: rknn-toolkit2 not installed.")
    print("Install with: pip install rknn-toolkit2")
    sys.exit(1)


# ONNX model definitions with fixed shapes for RKNN conversion.
# Input names match the actual ONNX models from sample_inference.py.
#
# Config reference (config.json):
#   talker: hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
#           num_key_value_heads=8, head_dim=128, vocab_size=3072
#   code_predictor: hidden_size=1024, num_hidden_layers=5, num_code_groups=16
#   speaker_encoder: enc_dim=1024, sample_rate=24000
#   text_hidden_size=2048, text_vocab_size=151936
MODELS = [
    {
        "name": "speaker_encoder",
        "onnx": "speaker_encoder_q.onnx",
        "rknn": "speaker_encoder_q.rknn",
        # Input: mel spectrogram (128 mel bins, 128 time frames)
        "inputs": [["mels", "float32", [1, 128, 128]]],
        "desc": "Speaker embedding extractor → [1, 1024]",
        "dual_core": False,
    },
    {
        "name": "text_project",
        "onnx": "text_project_q.onnx",
        "rknn": "text_project_q.rknn",
        # Input: text token IDs (dynamic → fixed at 512)
        "inputs": [["input_ids", "int64", [1, 512]]],
        "desc": "Text token projector → [1, 512, 1024]",
        "dual_core": False,
    },
    {
        "name": "tokenizer12hz_encode",
        "onnx": "tokenizer12hz_encode_q.onnx",
        "rknn": "tokenizer12hz_encode_q.rknn",
        # Input: raw audio (1s=24000 samples) + padding mask
        "inputs": [
            ["input_values", "float32", [1, 24000]],
            ["padding_mask", "int64", [1, 24000]],
        ],
        "desc": "Audio codec encoder → audio_codes [1, n_codebooks, n_frames]",
        "dual_core": False,
    },
    {
        "name": "tokenizer12hz_decode",
        "onnx": "tokenizer12hz_decode_q.onnx",
        "rknn": "tokenizer12hz_decode_q.rknn",
        # Input: audio codes [batch, n_codebooks, n_frames]
        # 16 code groups, ~128 frames for ~10s of audio at 12Hz
        "inputs": [["audio_codes", "int64", [1, 16, 128]]],
        "desc": "Audio codec decoder (vocoder) → 24kHz PCM",
        "dual_core": True,
    },
    {
        "name": "codec_embed",
        "onnx": "codec_embed_q.onnx",
        "rknn": "codec_embed_q.rknn",
        # Input: codec token IDs (single token or short sequence)
        "inputs": [["input_ids", "int64", [1, 1]]],
        "desc": "Codec token embedder → [1, 1, 1024]",
        "dual_core": False,
    },
    {
        "name": "code_predictor_embed",
        "onnx": "code_predictor_embed_q.onnx",
        "rknn": "code_predictor_embed_q.rknn",
        # Input: code ID + generation step
        "inputs": [
            ["input_ids", "int64", [1, 1]],
            ["generation_step", "int64", [1]],
        ],
        "desc": "Code predictor embedding → [1, 1, 1024]",
        "dual_core": False,
    },
    {
        "name": "code_predictor",
        "onnx": "code_predictor_q.onnx",
        "rknn": "code_predictor_q.rknn",
        # Input: hidden states + generation step
        "inputs": [
            ["inputs_embeds", "float32", [1, 1, 1024]],
            ["generation_step", "int64", [1]],
        ],
        "desc": "Sub-codebook predictor (5 layers) → logits [1, 1, 2048]",
        "dual_core": False,
    },
    {
        "name": "talker_prefill",
        "onnx": "talker_prefill_q.onnx",
        "rknn": "talker_prefill_q.rknn",
        # Input: embeddings + attention mask (dynamic → fixed at 512)
        "inputs": [
            ["inputs_embeds", "float32", [1, 512, 1024]],
            ["attention_mask", "int64", [1, 512]],
        ],
        "desc": "Talker prefill (28 layers) → logits + KV cache",
        "dual_core": True,
        "notes": "Largest model. KV cache: 28 layers × 2 (K,V) × 8 heads × 128 dim",
    },
    {
        "name": "talker_decode",
        "onnx": "talker_decode_q.onnx",
        "rknn": "talker_decode_q.rknn",
        # Input: single token embed + attention mask + KV cache
        # KV cache: 28 layers × 2 (key,value) = 56 tensors
        #   each: [1, 8, seq_len, 128] (num_kv_heads=8, head_dim=128)
        "inputs": [
            ["inputs_embeds", "float32", [1, 1, 1024]],
            ["attention_mask", "int64", [1, 513]],  # prefill(512) + 1 decode step
            # KV cache inputs auto-detected from ONNX model
        ],
        "desc": "Talker decode (autoregressive, batch=1 seq=1) + KV cache IO",
        "dual_core": True,
        "notes": "KV cache tensors auto-detected. May fail RKNN conversion due to dynamic shapes.",
    },
]

HF_REPO = "sivasub987/Qwen3-TTS-0.6B-ONNX-INT8"


def download_models(output_dir):
    """Download ONNX models from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading models from {HF_REPO}...")
    local_dir = snapshot_download(
        repo_id=HF_REPO,
        local_dir=os.path.join(output_dir, "onnx_src"),
        allow_patterns=["*.onnx", "*.json", "*.txt"],
    )
    print(f"Downloaded to: {local_dir}")
    return local_dir


def inspect_onnx(onnx_path):
    """Print ONNX model input/output info."""
    try:
        import onnx
    except ImportError:
        print("  (install onnx for model inspection: pip install onnx)")
        return

    model = onnx.load(onnx_path)
    print(f"  ONNX inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"    {inp.name}: shape={shape} dtype={dtype}")
    print(f"  ONNX outputs:")
    for out in model.graph.output[:5]:  # First 5 outputs
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: shape={shape}")
    if len(model.graph.output) > 5:
        print(f"    ... +{len(model.graph.output) - 5} more outputs")


def convert_one(onnx_path, rknn_path, input_shapes, target_platform="rk3576"):
    """Convert a single ONNX model to RKNN."""
    rknn = RKNN()

    # Config: INT8 quantization already done in ONNX, keep as-is
    rknn.config(
        target_platform=target_platform,
        optimization_level=3,
    )

    # Load ONNX
    print(f"  Loading ONNX: {onnx_path}")
    input_size_list = [spec[2] for spec in input_shapes]
    ret = rknn.load_onnx(model=onnx_path, input_size_list=input_size_list)
    if ret != 0:
        print(f"  ERROR: load_onnx failed (ret={ret})")
        rknn.release()
        return False

    # Build
    print(f"  Building RKNN...")
    ret = rknn.build(do_quantization=False)  # Already INT8
    if ret != 0:
        print(f"  ERROR: build failed (ret={ret})")
        rknn.release()
        return False

    # Export
    print(f"  Exporting: {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"  ERROR: export failed (ret={ret})")
        rknn.release()
        return False

    rknn.release()
    size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"  OK ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS ONNX INT8 → RKNN")
    parser.add_argument("--onnx-dir", help="Directory containing ONNX models (skip download)")
    parser.add_argument("--output-dir", default="./tts_rknn_models", help="Output directory")
    parser.add_argument("--target", default="rk3576", help="Target platform (default: rk3576)")
    parser.add_argument("--model", help="Convert only this model (by name)")
    parser.add_argument("--inspect", action="store_true", help="Only inspect ONNX models, don't convert")
    parser.add_argument("--auto-shape", action="store_true",
                        help="Auto-detect input shapes from ONNX (use with --inspect first)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Download or use existing ONNX models
    if args.onnx_dir:
        onnx_dir = args.onnx_dir
    else:
        onnx_dir = download_models(args.output_dir)

    # Filter models
    models = MODELS
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"ERROR: Unknown model '{args.model}'. Available: {[m['name'] for m in MODELS]}")
            sys.exit(1)

    # Inspect-only mode
    if args.inspect:
        for model in models:
            onnx_path = os.path.join(onnx_dir, model["onnx"])
            print(f"\n{'='*60}")
            print(f"{model['name']} — {model['desc']}")
            print(f"{'='*60}")
            if os.path.exists(onnx_path):
                size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"  File: {onnx_path} ({size_mb:.1f} MB)")
                inspect_onnx(onnx_path)
            else:
                print(f"  NOT FOUND: {onnx_path}")
        return

    # Convert each model
    results = {}
    for model in models:
        print(f"\n{'='*60}")
        print(f"Converting: {model['name']} — {model['desc']}")
        if "notes" in model:
            print(f"  Note: {model['notes']}")
        print(f"{'='*60}")

        onnx_path = os.path.join(onnx_dir, model["onnx"])
        rknn_path = os.path.join(args.output_dir, model["rknn"])

        if not os.path.exists(onnx_path):
            print(f"  WARNING: ONNX file not found: {onnx_path}")
            results[model["name"]] = "SKIP (not found)"
            continue

        # Show ONNX info
        inspect_onnx(onnx_path)

        try:
            ok = convert_one(onnx_path, rknn_path, model["inputs"], args.target)
            results[model["name"]] = "OK" if ok else "FAILED"
        except Exception as e:
            print(f"  ERROR: {e}")
            results[model["name"]] = f"ERROR ({e})"

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "+" if status == "OK" else "!"
        print(f"  [{icon}] {name}: {status}")

    failed = [n for n, s in results.items() if s != "OK" and not s.startswith("SKIP")]
    if failed:
        print(f"\nFailed models: {failed}")
        print("These will use onnxruntime CPU fallback on device.")
        print()
        print("Expected issues:")
        print("  - talker_decode: dynamic KV cache shapes may fail RKNN conversion")
        print("  - speaker_encoder/tokenizer12hz_*: ConvInteger ops may need workaround")
        print()
        print("Workaround: keep failed models as ONNX, run via onnxruntime on device CPU.")

    ok_count = sum(1 for s in results.values() if s == "OK")
    print(f"\nConverted: {ok_count}/{len(results)}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Next steps:")
    print("  1. Upload RKNN models to HuggingFace")
    print("  2. Copy ONNX fallback models for failed conversions")
    print("  3. Run: bash devtools/setup_tts.sh")


if __name__ == "__main__":
    main()
