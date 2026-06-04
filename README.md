# Minecraft Object Detection

Object detection project for Minecraft screenshots and gameplay footage based on MMDetection.

## Repository Structure

```text
.
├── configs/               # Training and model configs
├── datasets/              # Dataset files (not tracked by git)
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
./install.sh cpu
```

### CUDA Environment

```bash
./install.sh gpu
```

Verify installation:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
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
