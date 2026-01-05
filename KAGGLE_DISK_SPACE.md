# Kaggle Disk Space Management

## Problem
Kaggle notebooks have limited disk space (~30GB total, with ~5-15GB available after system files). When downloading datasets, you may encounter:
```
RuntimeError: Data processing error: CAS service error : IO Error: No space left on device (os error 28)
```

## Solutions

### 1. Use Streaming Mode (Recommended)
The code now automatically uses streaming mode when `OWT_SUBSET_NUM` is set. This downloads data on-the-fly without filling up disk.

```python
# In your Kaggle notebook
import os
os.environ["OWT_SUBSET_NUM"] = "11000"  # Only streams 11k examples, ~100MB
```

### 2. Clean Cache Before Training
Run the cleanup script to remove old cached files:

```bash
bash scripts/cleanup_kaggle_cache.sh
```

This removes:
- `/kaggle/working/hf` (default HuggingFace cache)
- `/kaggle/working/duo/duo/duo/duo/outputs` (old training outputs)

### 3. Monitor Disk Usage
Check available space:
```bash
df -h /kaggle/working
```

### 4. Recommended Kaggle Training Command
```bash
# Clean up first
bash scripts/cleanup_kaggle_cache.sh

# Set small subset (11k examples = ~100MB download)
export OWT_SUBSET_NUM=11000

# Train with minimal resources
python main.py \
    mode=train \
    data=openwebtext-split \
    algo=duo_base \
    algo.backbone=hf_dit \
    strategy=single \
    data.cache_dir=/kaggle/working/hf_cache \
    loader.batch_size=2 \
    loader.global_batch_size=2 \
    loader.eval_batch_size=2 \
    loader.eval_global_batch_size=2 \
    model.length=512 \
    trainer.max_steps=100 \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.accelerator=gpu \
    trainer.precision=16 \
    trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=10 \
    +wandb.offline=true
```

## How the Streaming Fix Works

### Before (Disk Space Issue)
```python
# This downloads ENTIRE dataset first (~40GB), then slices
dataset = datasets.load_dataset('openwebtext', split='train[:11000]', streaming=False)
```

### After (Streaming Mode)
```python
# This streams only the first 11k examples (~100MB total download)
dataset = datasets.load_dataset('openwebtext', split='train', streaming=True)
dataset = dataset.take(11000)
dataset = datasets.Dataset.from_dict({k: [item[k] for item in dataset] for k in ['text']})
```

## Disk Space Requirements by Subset Size

| `OWT_SUBSET_NUM` | Approx. Download Size | Recommended For |
|------------------|----------------------|-----------------|
| 1000 | ~10 MB | Quick testing |
| 11000 | ~100 MB | Initial training (default in scripts) |
| 50000 | ~500 MB | Longer training |
| 100000 | ~1 GB | Extended training |
| Full dataset (no env var) | ~40 GB | ❌ Not recommended on Kaggle |

## Troubleshooting

### Still Getting Disk Errors?
1. **Restart the Kaggle kernel** to clear all caches
2. **Run cleanup script immediately** after restart
3. **Use smaller subset**: `export OWT_SUBSET_NUM=1000`
4. **Check your Kaggle session**: Make sure "Internet" is enabled in notebook settings

### Multiple Cache Directories
The code uses `/kaggle/working/hf_cache`, but HuggingFace may also create `/kaggle/working/hf`. The cleanup script removes both.

### Already Downloaded Partial Files?
If the download was interrupted, you may have partial files taking up space:
```bash
# Remove all HuggingFace caches
rm -rf /kaggle/working/hf*

# Then pull latest code and try again
cd /kaggle/working/duo/duo
git pull origin main
```
