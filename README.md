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
# Basic training
python ai-model/train.py --epochs 60 --batch_size 32

# Recommended for best accuracy (add --balance if classes are imbalanced)
python ai-model/train.py --epochs 60 --batch_size 32 --balance --unfreeze_layer1
```

If you want a faster/smaller model for quick experimentation, add `--fast` to train ResNet‑18:
```powershell
python ai-model/train.py --fast --epochs 50 --batch_size 64
```

## Training Guide for 90–95% Accuracy

### Optimal Training Command
```bash
python ai-model/train.py \
  --epochs 60 \
  --batch_size 32 \
  --lr 0.0005 \
  --balance \
  --unfreeze_layer1
```

### Key Training Arguments
| Argument | Default | Description |
|---|---|---|
| `--epochs` | 60 | Number of training epochs |
| `--lr` | 0.0003 | Learning rate |
| `--batch_size` | 32 | Batch size (auto-capped to 4 on CPU for ResNet-101) |
| `--balance` | off | Use weighted sampler to handle class imbalance |
| `--unfreeze_layer1` | off | Unfreeze backbone layer1 for finer mineral features |
| `--mixup_alpha` | 0.2 | Mixup augmentation alpha (0 to disable) |
| `--clip_grad_norm` | 1.0 | Gradient clipping max norm (0 to disable) |
| `--warmup_epochs` | 5 | Linear LR warmup epochs before cosine annealing |

### Implemented Improvements
- **Enhanced data augmentation**: GaussianBlur, RandomRotation (25°), RandomAffine, Mixup (α=0.2)
- **Larger FC head**: 2048 → 1024 → 512 → num_classes with BatchNorm for stability
- **Reduced dropout**: 0.4 → 0.25 for better feature flow
- **Warmup scheduler**: 5-epoch linear warmup → CosineAnnealingLR for faster convergence
- **Gradient clipping**: max_norm=1.0 prevents training instability
- **Lower label smoothing**: 0.1 → 0.05 for harder learning signals
- **Higher patience**: 10 → 15 for more stable early stopping
- **Optional layer1 unfreezing**: capture low-level mineral texture features

### Hardware Recommendations
| Setup | Command |
|---|---|
| GPU (8 GB+ VRAM) | `python ai-model/train.py --epochs 60 --batch_size 32 --balance --unfreeze_layer1` |
| GPU (4 GB VRAM) | `python ai-model/train.py --epochs 60 --batch_size 16 --balance` |
| CPU only (quick test) | `python ai-model/train.py --fast --epochs 50 --batch_size 32 --balance` |

### Troubleshooting Low Accuracy
1. **Enable `--balance`**: If some mineral classes dominate the dataset, weighted sampling improves per-class accuracy.
2. **Enable `--unfreeze_layer1`**: Unlocking layer1 lets the model capture mineral-specific texture features at the cost of slightly longer training.
3. **Increase epochs**: Use `--epochs 80` if the model hasn't converged yet.
4. **Lower learning rate**: Try `--lr 0.0001` if training loss oscillates.
5. **Check data quality**: Ensure each mineral class has at least 100–200 images.
6. **Disable Mixup**: If accuracy is unusually low, try `--mixup_alpha 0` to rule out Mixup causing issues with small datasets.

## Evaluate
```powershell
python ai-model/evaluate.py \
  --model_path ai-model/models/resnet101_mineral.pth \
  --data_dir ai-model/dataset/test
```

Outputs:
- Overall accuracy and macro/weighted F1-scores
- Per-class accuracy and F1-score table
- Full sklearn classification report (precision, recall, F1 per class)
- Confusion matrix image saved to `confusion_matrix.png`

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
---
app.py` so your web UI shows detected objects automatically.
