# SmartMine Vision Lab (Flask)

## Setup (Windows)
1. Create & activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Train (ResNet‑101)
The default training script uses **ResNet-101** (recommended for best accuracy). To train:
```powershell
python ai-model/train.py --epochs 50 --batch_size 32
```

### Recommended settings for best accuracy
Use the following command to get the best results. It enables balanced sampling,
layer1 unfreezing, Mixup augmentation, gradient clipping, and a 5-epoch warmup:
```powershell
python ai-model/train.py --epochs 50 --batch_size 32 --balance --unfreeze_layer1 --mixup_alpha 0.2 --clip_grad_norm 1.0 --warmup_epochs 5
```

### GPU training (NVIDIA)
For systems with a CUDA-capable GPU the script automatically enables mixed
precision (AMP) for faster training:
```powershell
python ai-model/train.py --epochs 50 --batch_size 64 --balance --unfreeze_layer1 --mixup_alpha 0.2
```

### CPU-only / low memory machines
ResNet-101 is memory-heavy on CPU. The script auto-caps the batch size to 4,
but you can also use the lightweight ResNet-18 variant (5× faster):
```powershell
python ai-model/train.py --fast --epochs 20 --batch_size 32
```

### All available training flags
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Total training epochs |
| `--batch_size` | 32 | Mini-batch size |
| `--lr` | 0.0003 | Initial learning rate |
| `--balance` | off | Weighted sampler to balance classes |
| `--unfreeze_layer1` | off | Also unfreeze ResNet layer1 |
| `--mixup_alpha` | 0.2 | Mixup alpha (0 to disable) |
| `--clip_grad_norm` | 1.0 | Gradient clipping max-norm (0 to disable) |
| `--warmup_epochs` | 5 | Linear LR warmup epochs |
| `--randaugment` | off | Enable RandAugment policy |
| `--fast` | off | Use ResNet-18 instead of ResNet-101 |
| `--data_dir` | `ai-model/dataset_balanced` | Dataset root directory |

## Evaluate
After training, run the evaluation script to get per-class accuracy and F1-scores:
```powershell
python ai-model/evaluate.py --model_path ai-model/models/resnet101_mineral.pth --data_dir ai-model/dataset/test
```

## Run the web UI
```powershell
python flask_mineral_app.py
```

## UI
You can see the UI by opening:

`http://localhost:5002`

### Notes
- The Flask app prefers the **ResNet-101** checkpoint file (`ai-model/models/resnet101_mineral.pth`).
- If you want to force a different checkpoint, set `MINERAL_MODEL_PATH` before starting the app.

## Troubleshooting low accuracy

| Symptom | Fix |
|---------|-----|
| Validation accuracy plateaus early | Increase `--epochs`, lower `--lr`, or add `--unfreeze_layer1` |
| One class dominates training | Use `--balance` for weighted sampling |
| High training / low validation accuracy (overfitting) | Increase dropout, add `--randaugment`, reduce `--epochs` |
| GPU out-of-memory | Reduce `--batch_size` (try 16 or 8) |
| CPU very slow | Switch to `--fast` (ResNet-18) for experiments |
| Loss stays flat at start | Increase `--warmup_epochs` to 10 |
