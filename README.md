# Arrange Training Reproduction Guide

The main training entry point in this project is:

- `scripts/train_iter/train_diffusion.py`

This document provides a minimal end-to-end workflow to reproduce training from scratch.

## 1. Environment Setup

Recommended environment (adjust as needed):

- Python 3.9+
- PyTorch (matching your CUDA version)
- GPU (optional, CPU also works but is slower)

Move to the project root (the directory containing `Arrange`):

```powershell
cd d:\pythonhomework\Github\ArrangeBot\Arrange
```

Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If your CUDA version is different, use the matching PyTorch install command.

## 2. Data and Configuration

The training script reads:

- `config/config_scene.yaml`

Most important fields are:

- `root.raw`
- `root.file_path`

Update them to local paths on your machine, for example:

```yaml
root:
  raw: "D:/your_data/raw"
  file_path: "D:/your_data/train_scenes.txt"
```

> Note: the current paths in this file point to the original author's server (for example `/remote-home/...`) and usually cannot be used directly on local machines.

## 3. Run Training

Run from the parent directory of `Arrange` (to keep relative imports/path behavior consistent with the script):

```powershell
cd d:\pythonhomework\Github\ArrangeBot
python .\Arrange\scripts\train_iter\train_diffusion.py --config_file .\Arrange\config\config_scene.yaml --n_processes 0 --seed 27
```

Common arguments:

- `--config_file`: config path, default is `./Arrange/config/config_scene.yaml`
- `--n_processes`: `num_workers` for DataLoader
- `--seed`: random seed

## 4. Output Behavior

During training, batch loss values are printed.

Current checkpoint logic in the script:

- Check every `100` epochs
- Save weights when `epoch_loss` is better than `best_loss`
- Output folder name format is `Arrange_HOUR-MINUTE`

## 5. Troubleshooting

### 5.1 Path not found errors

Check first:

- `root.raw` exists in `config/config_scene.yaml`
- `root.file_path` exists and has valid content format

### 5.2 `ModuleNotFoundError` or relative import issues

Run from the parent directory of `Arrange` (see Section 3), otherwise `sys.path.append('./Arrange')` may not resolve correctly.

### 5.3 Error when creating output folder

There is a potential typo in `train_diffusion.py`:

- `now.mi` should be `now.minute`

If you see time-related errors, change:

```python
folder_name = f"Arrange_{now.hour}-{now.mi}"
```

to:

```python
folder_name = f"Arrange_{now.hour}-{now.minute}"
```

## 6. Minimal Reproduction Checklist

Before training, confirm:

- Virtual environment is activated
- `torch/numpy/tqdm/onnx/pyyaml` are installed
- Local data paths are set in `config/config_scene.yaml`
- Training command is executed from the parent directory of `Arrange`
- The first run prints loss values and creates an output folder successfully


