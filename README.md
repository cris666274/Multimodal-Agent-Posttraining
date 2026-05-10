# multimodal-agent-posttraining

## Environment

Create and install the local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Activate it for interactive work:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Smoke Checks

Check Python package imports:

```powershell
.\.venv\Scripts\python -c "import torch, torchvision, torchaudio, accelerate, transformers, datasets, gradio, rich; import qwen_vl_utils; print('ok')"
```

Check seed data and image paths:

```powershell
.\.venv\Scripts\python scripts\check_seed_data.py
```
