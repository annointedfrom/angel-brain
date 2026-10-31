#!/usr/bin/env bash
# Manual HF-merged -> GGUF q4_k_m, bypassing Unsloth's flaky auto-builder.
# The merged 16-bit model already exists in angelbrain_gguf/ (from export_gguf merge step).
set -e
cd ~/training

echo "[1/5] clone llama.cpp (shallow) if missing"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp

echo "[2/5] ensure python deps for the converter"
./.venv/bin/pip install -q gguf sentencepiece 2>&1 | tail -1 || true

echo "[3/5] convert merged HF -> f16 GGUF"
./.venv/bin/python llama.cpp/convert_hf_to_gguf.py angelbrain_gguf \
    --outfile angelbrain.f16.gguf --outtype f16

echo "[4/5] build llama-quantize (cmake)"
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON > /tmp/cmake_cfg.log 2>&1
cmake --build llama.cpp/build --target llama-quantize -j4 > /tmp/cmake_build.log 2>&1

echo "[5/5] quantize f16 -> q4_k_m"
QBIN=$(find llama.cpp/build -name 'llama-quantize' -type f | head -1)
"$QBIN" angelbrain.f16.gguf angelbrain.q4_k_m.gguf q4_k_m

echo "DONE. Artifacts:"
ls -la ~/training/angelbrain*.gguf
