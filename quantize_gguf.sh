#!/usr/bin/env bash
# quantize_gguf.sh <merged_hf_dir> <name> - merged 16-bit HF dir -> <name>.q4_k_m.gguf (WSL).
# Generalizes convert_gguf_v2.sh. Reuses the llama.cpp build in ~/training (builds it if missing).
set -euo pipefail
MERGED=${1:?merged HF dir}
NAME=${2:?output name}
cd ~/training

echo "[1/3] convert $MERGED -> $NAME.f16.gguf"
./.venv/bin/python llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$NAME.f16.gguf" --outtype f16

echo "[2/3] locate llama-quantize"
QBIN=$(find llama.cpp/build -name 'llama-quantize' -type f | head -1)
if [ -z "$QBIN" ]; then
    cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON > /tmp/cmake_cfg.log 2>&1
    cmake --build llama.cpp/build --target llama-quantize -j4 > /tmp/cmake_build.log 2>&1
    QBIN=$(find llama.cpp/build -name 'llama-quantize' -type f | head -1)
fi

echo "[3/3] quantize -> $NAME.q4_k_m.gguf"
"$QBIN" "$NAME.f16.gguf" "$NAME.q4_k_m.gguf" q4_k_m
rm -f "$NAME.f16.gguf"
ls -la "$NAME.q4_k_m.gguf"
