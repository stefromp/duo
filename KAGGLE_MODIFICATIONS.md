# Kaggle Modifications Summary

This fork includes modifications to run DUO on Kaggle with limited resources.

## What's Modified

### 1. `dataloader.py` - OpenWebText Subset Loading
Added support for loading a subset of OpenWebText via environment variable:

```python
# Before: Loads full 8M documents (~40GB)
dataset = datasets.load_dataset('openwebtext', split='train[:-100000]')

# After: Loads only specified subset
export OWT_SUBSET_NUM=11000  # 9,900 train + 1,100 val
dataset = datasets.load_dataset('openwebtext', split='train[:9900]')
```

**Benefits:**
- Downloads only ~100MB instead of 40GB
- 90/10 train/validation split
- Deterministic and reproducible
- Backward compatible (works without env var)

### 2. `models/dit.py` - FlashAttention Optional
Made FlashAttention import optional with automatic fallback:

```python
# Before: Hard dependency
import flash_attn

# After: Optional with fallback
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    # Falls back to PyTorch SDPA
```

**Benefits:**
- No installation errors on Kaggle T4/P100 GPUs
- Automatic fallback to PyTorch's efficient SDPA
- Minimal performance impact
- Works everywhere

## New Files for Kaggle

- `kaggle_train.py` - Automated training script
- `test_kaggle_setup.py` - Setup verification
- `KAGGLE_SETUP.md` - Quick setup guide
- `scripts/train_owt_kaggle.sh` - Training script
- `scripts/eval_owt_kaggle.sh` - Evaluation script
- `scripts/gen_owt_kaggle.sh` - Generation script
- `kaggle_notebook.ipynb` - Jupyter notebook template

## Quick Start on Kaggle

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git duo
%cd duo
!pip install -q transformers datasets pytorch-lightning hydra-core omegaconf wandb

import os
os.environ['OWT_SUBSET_NUM'] = '11000'

!python main.py mode=train data=openwebtext-split algo=duo_base \
  loader.batch_size=2 model.length=512 trainer.max_steps=500 \
  trainer.devices=1 trainer.precision=16 +wandb.offline=true
```

See [KAGGLE_SETUP.md](KAGGLE_SETUP.md) for detailed instructions.

## Testing Locally

```bash
# Verify modifications
python test_kaggle_setup.py

# Test with small subset
export OWT_SUBSET_NUM=1000
./scripts/train_owt_kaggle.sh
```

## Credits

Based on the original DUO repository:
- Paper: https://arxiv.org/abs/2406.07524
- Original repo: https://github.com/kuleshov-group/discrete-diffusion-guidance
- Model: https://huggingface.co/exdysa/duo

Modifications for Kaggle compatibility by [Your Name].
