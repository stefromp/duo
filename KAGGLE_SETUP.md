# Running DUO on Kaggle - Quick Setup Guide

This repository has been modified to run efficiently on Kaggle with limited resources.

## 🚀 Quick Start for Kaggle

### Option 1: Using Kaggle Notebook (Recommended)

1. **Create a new Kaggle Notebook**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Enable **GPU** accelerator (T4, P100, or A100)
   - Enable **Internet** access

2. **Clone this repository**
   ```python
   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git duo
   %cd duo
   ```

3. **Install dependencies**
   ```python
   !pip install -q transformers datasets pytorch-lightning hydra-core omegaconf wandb fsspec
   ```

4. **Set configuration and train**
   ```python
   import os
   os.environ['OWT_SUBSET_NUM'] = '11000'  # Use 11k documents
   
   !python main.py \
     mode=train \
     data=openwebtext-split \
     algo=duo_base \
     algo.backbone=hf_dit \
     loader.batch_size=2 \
     loader.global_batch_size=2 \
     model.length=512 \
     trainer.max_steps=500 \
     trainer.devices=1 \
     trainer.precision=16 \
     +wandb.offline=true
   ```

### Option 2: Using the Automated Script

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git duo
%cd duo
!python kaggle_train.py
```

## 📝 Key Modifications

This repository includes two critical patches for Kaggle compatibility:

### 1. OpenWebText Subset Loading (`dataloader.py`)
- Control dataset size via `OWT_SUBSET_NUM` environment variable
- Downloads only requested subset instead of full 40GB dataset
- 90/10 train/validation split

**Usage:**
```python
os.environ['OWT_SUBSET_NUM'] = '11000'  # 9,900 train + 1,100 val
```

### 2. FlashAttention Optional (`models/dit.py`)
- Automatic fallback to PyTorch SDPA when FlashAttention unavailable
- No installation errors on Kaggle GPUs
- Minimal performance impact

**No action needed** - works automatically!

## ⚙️ Configuration Options

### Subset Sizes
```python
os.environ['OWT_SUBSET_NUM'] = '1000'    # Quick test (5 min)
os.environ['OWT_SUBSET_NUM'] = '11000'   # Standard (15 min) ✅ Recommended
os.environ['OWT_SUBSET_NUM'] = '50000'   # Extended (2 hours)
```

### Training Parameters

| Parameter | Recommended | Memory Limited | Large GPU |
|-----------|-------------|----------------|-----------|
| `loader.batch_size` | 2 | 1 | 4 |
| `model.length` | 512 | 256 | 1024 |
| `trainer.max_steps` | 500 | 100 | 5000 |
| `trainer.precision` | 16 | 16 | 16 |

## 🔧 Training Scripts

Pre-configured shell scripts are available in `scripts/`:

```bash
# Training
./scripts/train_owt_kaggle.sh

# Evaluation
./scripts/eval_owt_kaggle.sh

# Text generation
./scripts/gen_owt_kaggle.sh
```

## ✅ Verification

Before training, verify the setup:

```python
!python test_kaggle_setup.py
```

Expected output:
```
✓ All tests passed!
You're ready to run training on Kaggle!
```

## 📊 Expected Results (T4 GPU, Default Settings)

- **Speed**: 2-3 seconds/step
- **Memory**: 4-6 GB GPU RAM
- **Training time**: ~15 minutes (500 steps)
- **Download size**: ~100 MB (11k subset)
- **Checkpoint size**: ~500 MB

## 🐛 Troubleshooting

### Out of Memory
```python
# Use smaller resources
os.environ['OWT_SUBSET_NUM'] = '1000'
# Then in training command:
loader.batch_size=1 model.length=256
```

### FlashAttention Import Error
**No action needed!** The code automatically falls back to PyTorch SDPA.

### Dataset Download Slow
```python
# Start with smaller subset
os.environ['OWT_SUBSET_NUM'] = '1000'
```

## 📁 Important Files

- `dataloader.py` - Modified for subset loading
- `models/dit.py` - Modified for optional FlashAttention
- `kaggle_train.py` - Automated training script
- `test_kaggle_setup.py` - Setup verification
- `scripts/train_owt_kaggle.sh` - Training shell script
- `scripts/eval_owt_kaggle.sh` - Evaluation shell script
- `scripts/gen_owt_kaggle.sh` - Generation shell script

## 🎓 Full Training Command

```bash
export OWT_SUBSET_NUM=11000

python main.py \
  mode=train \
  data=openwebtext-split \
  algo=duo_base \
  algo.backbone=hf_dit \
  loader.batch_size=2 \
  loader.global_batch_size=2 \
  loader.eval_batch_size=2 \
  model.length=512 \
  trainer.max_steps=500 \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.precision=16 \
  trainer.log_every_n_steps=50 \
  trainer.val_check_interval=0.2 \
  +wandb.offline=true
```

## 🔗 Resources

- **Original DUO Paper**: https://arxiv.org/abs/2406.07524
- **Original Repository**: https://github.com/kuleshov-group/discrete-diffusion-guidance
- **HuggingFace Model**: https://huggingface.co/exdysa/duo
- **OpenWebText Dataset**: https://huggingface.co/datasets/openwebtext

## 💡 Pro Tips

1. **Start small**: Test with `OWT_SUBSET_NUM=1000` first (5 min)
2. **Monitor resources**: Use `!nvidia-smi` to check GPU usage
3. **Save checkpoints**: Copy to `/kaggle/working/` for persistence
4. **Clear cache**: Run `!rm -rf ~/.cache/huggingface/` between experiments

## 📋 Kaggle Notebook Template

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git duo
%cd duo
!pip install -q transformers datasets pytorch-lightning hydra-core omegaconf wandb

# Cell 2: Verify
!python test_kaggle_setup.py

# Cell 3: Configure
import os
os.environ['OWT_SUBSET_NUM'] = '11000'
os.environ['WANDB_MODE'] = 'offline'

# Cell 4: Train
!./scripts/train_owt_kaggle.sh

# Cell 5: Evaluate (optional)
!./scripts/eval_owt_kaggle.sh

# Cell 6: Generate samples (optional)
!./scripts/gen_owt_kaggle.sh
```

## ⚠️ Important Notes

- This is a **proof-of-concept** setup for testing DUO on Kaggle
- Full training requires much more data and compute time
- Results with 500 steps will show minimal improvement (expected)
- For production use, train with full dataset on multi-GPU setup

---

**Ready to train?** Follow the Quick Start section above!

For issues or questions, please open an issue on GitHub.
