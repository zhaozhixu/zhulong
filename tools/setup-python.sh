#!/bin/sh
# Create the project venv (uv-managed Python) and install the pinned torch
# plus the packages reference generation needs. One torch serves both
# roles: the Python side (reference generation, the future compilation
# sidecar) and the libtorch dylibs that runtime/ffi dlopens — a single
# installation, so the two sides cannot drift apart. Idempotent.
set -eu

PY=3.14
TORCH=2.13.0

cd "$(dirname "$0")/.."
[ -d .venv ] || uv venv --python "$PY"
uv pip install --quiet "torch==$TORCH" numpy safetensors
.venv/bin/python -c "import torch, sys; print('venv ready: torch', torch.__version__, '| python', sys.version.split()[0])"
