#!/bin/bash
set -e

echo "Extracting code..."
tar -xzf project_code.tar.gz

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "Setting up Python 3.12 and environment..."
uv python install 3.12
uv sync

echo "Installing JAX with TPU support..."
# Force install jax[tpu] into the venv
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "Downloading Gemma 4 model weights..."
# Pass the token via env var
export HUGGING_FACE_TOKEN=$1
uv run download_model.py

echo "Starting training..."
# JAX memory optimizations
export JAX_PLATFORMS=tpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run train_tpu.py
