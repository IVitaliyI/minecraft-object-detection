#!/bin/bash
set -e

ENV_TYPE=${1:-cpu}

echo "=== 1. Синхронизация базового окружения ($ENV_TYPE) ==="
uv sync --extra "$ENV_TYPE"

if [ "$ENV_TYPE" = "gpu" ]; then
  echo "=== 2. Установка MMCV для GPU ==="
  wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.1.0.tar.gz
  tar -xvf v2.1.0.tar.gz
  cd mmcv-2.1.0
  uv pip install -r requirements/optional.txt
  CUDA_HOME=/usr/local/cuda-12.1 FORCE_CUDA=1 uv pip install -e . --no-build-isolation -v
  uv run python .dev_scripts/check_installation.py
  cd ..
else
  echo "=== 2. Установка MMCV для CPU ==="
  uv pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html \
    --no-build-isolation
fi

echo "=== 3. Установка локального mmdetection ==="
uv pip install ./mmdetection --no-build-isolation

echo "🎉 Окружение успешно настроено!"
