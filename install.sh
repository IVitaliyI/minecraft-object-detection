#!/bin/bash
set -e

ENV_TYPE=${1:-cpu}

echo "=== 1. Синхронизация базового окружения ($ENV_TYPE) ==="
uv sync --extra "$ENV_TYPE" --system-certs

if [ "$ENV_TYPE" = "gpu" ]; then
  echo "=== 2. Установка MMCV для GPU (CUDA 12.1) ==="
  uv pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    --system-certs --no-build-isolation
else
  echo "=== 2. Установка MMCV для CPU ==="
  uv pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html \
    --system-certs --no-build-isolation
fi

echo "=== 3. Установка локального mmdetection в editable-режиме ==="
uv --system-certs pip install -e ./mmdetection --no-build-isolation

echo "🎉 Окружение успешно настроено!"
