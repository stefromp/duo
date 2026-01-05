#!/bin/bash
# Training script for running DUO on Kaggle with OpenWebText subset
# This script is optimized for Kaggle's resource constraints:
# - Limited GPU memory (16GB on T4)
# - Limited disk space
# - Limited training time (9 hours max)

# Set the subset size (total number of documents)
# 11000 = ~9900 train + ~1100 validation
# Adjust this based on your needs and Kaggle's resources
export OWT_SUBSET_NUM=11000

# Optional: Set cache directory to control where datasets are stored
# export HF_DATASETS_CACHE="/kaggle/working/hf_cache"

echo "Starting DUO training on Kaggle with OpenWebText subset"
echo "Subset size: ${OWT_SUBSET_NUM} documents (~90% train, ~10% validation)"
echo "Expected: ~$((OWT_SUBSET_NUM * 9 / 10)) training examples, ~$((OWT_SUBSET_NUM / 10)) validation examples"

# Run training with Kaggle-optimized settings
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
  trainer.accelerator="gpu" \
  trainer.precision=16 \
  trainer.log_every_n_steps=50 \
  trainer.val_check_interval=0.2 \
  +trainer.enable_checkpointing=true \
  +trainer.callbacks.checkpoint.save_top_k=1 \
  +wandb.offline=true

echo "Training complete!"
echo "Check lightning_logs/ for checkpoints and logs"
