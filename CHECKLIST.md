# GitHub Upload & Kaggle Deployment Checklist

## 📤 Before Pushing to GitHub

### 1. Verify All Modifications Are in Place
```bash
# Check dataloader.py has subset support
grep "OWT_SUBSET_NUM" dataloader.py

# Check models/dit.py has FlashAttention optional
grep "FLASH_ATTN_AVAILABLE" models/dit.py

# Run verification script
python test_kaggle_setup.py
```

Expected output: `✓ All tests passed!`

### 2. Test Locally (Optional but Recommended)
```bash
# Quick test with tiny subset
export OWT_SUBSET_NUM=100
python main.py mode=train data=openwebtext-split algo=duo_base \
  loader.batch_size=1 model.length=256 trainer.max_steps=10 \
  trainer.devices=1 trainer.precision=16 +wandb.offline=true
```

### 3. Update GitHub Repository Info

Edit these files and replace placeholders:
- `KAGGLE_SETUP.md` - Line 13, 73: Replace `YOUR_USERNAME/YOUR_REPO_NAME`
- `KAGGLE_MODIFICATIONS.md` - Line 70, 87: Replace `YOUR_USERNAME/YOUR_REPO_NAME`

### 4. Add/Commit/Push to GitHub
```bash
cd /Users/stefanosrompos/Desktop/duo-main1

# Check status
git status

# Add all modified files
git add dataloader.py models/dit.py
git add kaggle_train.py test_kaggle_setup.py
git add scripts/train_owt_kaggle.sh scripts/eval_owt_kaggle.sh scripts/gen_owt_kaggle.sh
git add KAGGLE_SETUP.md KAGGLE_MODIFICATIONS.md kaggle_notebook.ipynb
git add CHECKLIST.md

# Commit
git commit -m "Add Kaggle compatibility: subset loading & optional FlashAttention"

# Push to your fork
git push origin main
```

---

## 🚀 Running on Kaggle

### Step 1: Create Kaggle Notebook
1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Settings:
   - ✅ Enable **GPU** accelerator (P100, T4, or A100)
   - ✅ Enable **Internet** access
   - ⚠️ Session timeout: Set to maximum (9 hours)

### Step 2: Clone Your Repository
```python
# In Kaggle notebook cell
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git duo
%cd duo
!ls -la  # Verify files are there
```

### Step 3: Install Dependencies
```python
!pip install -q transformers datasets pytorch-lightning hydra-core omegaconf wandb fsspec aiohttp
```

### Step 4: Verify Setup
```python
!python test_kaggle_setup.py
```

Expected output:
```
✓ PASS - Environment Variable
✓ PASS - Dataloader Modification
✓ PASS - Package Imports
✓ PASS - Split Calculation
✓ PASS - Config Files
✓ All tests passed!
```

### Step 5: Quick Test (5 minutes)
```python
import os
os.environ['OWT_SUBSET_NUM'] = '1000'

!python main.py mode=train data=openwebtext-split algo=duo_base \
  loader.batch_size=1 model.length=256 trainer.max_steps=100 \
  trainer.devices=1 trainer.precision=16 +wandb.offline=true
```

### Step 6: Full Training (15 minutes)
```python
import os
os.environ['OWT_SUBSET_NUM'] = '11000'

!./scripts/train_owt_kaggle.sh
```

### Step 7: Save Results
```python
# Copy checkpoint to output
import shutil
import glob

checkpoints = glob.glob('lightning_logs/*/checkpoints/*.ckpt')
if checkpoints:
    shutil.copy(checkpoints[-1], '/kaggle/working/final_model.ckpt')
    print(f"✓ Checkpoint saved to /kaggle/working/")
```

---

## ✅ Verification Checklist

### GitHub Upload
- [ ] `dataloader.py` modified with subset support
- [ ] `models/dit.py` modified with optional FlashAttention
- [ ] `kaggle_train.py` script added
- [ ] `test_kaggle_setup.py` script added
- [ ] Shell scripts in `scripts/` directory
- [ ] Documentation files added (`KAGGLE_SETUP.md`, etc.)
- [ ] Repository URLs updated in documentation
- [ ] Committed and pushed to GitHub

### Kaggle Setup
- [ ] Notebook created with GPU enabled
- [ ] Internet access enabled
- [ ] Repository cloned successfully
- [ ] Dependencies installed
- [ ] `test_kaggle_setup.py` passes
- [ ] Quick test runs without errors
- [ ] Full training completes
- [ ] Checkpoint saved

---

## 🐛 Common Issues & Solutions

### Issue: `git clone` fails
**Solution**: 
- Check repository is public
- Or use: `!git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO_NAME.git duo`

### Issue: Out of memory
**Solution**:
```python
os.environ['OWT_SUBSET_NUM'] = '1000'  # Smaller subset
# In training: loader.batch_size=1 model.length=256
```

### Issue: `test_kaggle_setup.py` fails
**Solution**:
- Check you're in the `duo` directory: `%cd duo`
- Verify files copied: `!ls dataloader.py models/dit.py`

### Issue: FlashAttention error
**Solution**: 
- Should work automatically (fallback enabled)
- If still errors, check `models/dit.py` has the modifications

### Issue: Dataset download slow
**Solution**:
```python
os.environ['OWT_SUBSET_NUM'] = '1000'  # Much smaller, faster download
```

---

## 📊 Expected Timeline

| Task | Time | Notes |
|------|------|-------|
| Create Kaggle notebook | 1 min | Enable GPU + Internet |
| Clone repository | 30 sec | Depends on repo size |
| Install dependencies | 5-10 min | First time only |
| Verify setup | 30 sec | Run test script |
| Quick test (1k subset, 100 steps) | 5 min | Verify everything works |
| Full training (11k subset, 500 steps) | 15 min | Recommended settings |
| Extended training (50k subset, 5k steps) | 2 hours | If time permits |

---

## 💡 Pro Tips

1. **Save Notebook**: Kaggle auto-saves, but click "Save Version" after training
2. **Monitor GPU**: Use `!nvidia-smi` to check memory usage
3. **Check Disk**: Use `!df -h` to monitor disk space
4. **Clear Cache**: Run `!rm -rf ~/.cache/huggingface/` if disk fills up
5. **Export Results**: Copy checkpoints and metrics to `/kaggle/working/`
6. **Use Sessions Wisely**: Kaggle has ~30 hours GPU/week limit

---

## 📞 Need Help?

If you encounter issues:

1. **Check the test script**: `!python test_kaggle_setup.py`
2. **Read error messages**: Usually point to the issue
3. **Check documentation**: See `KAGGLE_SETUP.md`
4. **Verify modifications**: Ensure both patches are applied
5. **Try smaller subset**: Start with `OWT_SUBSET_NUM=1000`

---

## 🎉 Success Indicators

You'll know it's working when you see:

```
✓ GPU detected
✓ Repository cloned
✓ All packages installed
✓ Dataloader supports subset loading
✓ Using OpenWebText subset: 9900 training examples
✓ Using OpenWebText subset: 1100 validation examples
✓ FlashAttention not available: ... (or available)
✓ Falling back to PyTorch native attention (SDPA)
✓ Training started
  Epoch 0:  10%|█         | 50/500 [02:30<22:30, 2.50s/it, loss=10.2]
```

Good luck with your training! 🚀
