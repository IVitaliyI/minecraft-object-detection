# Minecraft Object Detection

Object detection project for Minecraft screenshots and gameplay footage based on MMDetection.

## Repository Structure

```text
.
├── configs/               # Training and model configs
├── datasets/              # Dataset files (not tracked by git)
├── experiments/           # Training outputs
├── src/                   # Project source code
├── mmdetection/           # MMDetection git submodule
├── pyproject.toml
└── README.md
```

## Clone Repository

Clone the repository together with the MMDetection submodule:

```bash
git clone --recursive <repository-url>
```

If the repository has already been cloned:

```bash
git submodule update --init --recursive
```

## Environment Setup

### CPU Environment

```bash
uv sync --extra cpu
```

### CUDA Environment

```bash
uv sync --extra gpu
```

Verify installation:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output:

CPU:

```text
False
```

GPU:

```text
True
```

## Dataset Preparation

Prepare the dataset in COCO format:

```text
datasets/
└── minecraft/
    ├── train/
    ├── val/
    ├── annotations/
    │   ├── instances_train.json
    │   └── instances_val.json
```

## Training

Run training using MMDetection tools:

```bash
uv run python mmdetection/tools/train.py configs/minecraft/faster_rcnn.py
```

Save outputs to:

```text
work_dirs/
```

or configure a custom experiment directory in the config.

## Evaluation

```bash
uv run python mmdetection/tools/test.py \
    configs/minecraft/faster_rcnn.py \
    work_dirs/faster_rcnn/latest.pth
```

## Inference

```bash
uv run python demo/inference.py \
    --checkpoint work_dirs/faster_rcnn/latest.pth \
    --image example.png
```

## Updating MMDetection

Update the submodule:

```bash
cd mmdetection
git pull
cd ..
git add mmdetection
git commit -m "Update MMDetection"
```

## Development

Custom project code should be placed inside:

```text
src/
```

Avoid modifying MMDetection source code directly whenever possible. Use custom registries, datasets, hooks, transforms, and model components from the project source tree.

## References

* MMDetection
* MMEngine
* MMCV
* COCO Dataset Format

```
```
