#!/bin/bash
# Text generation/sampling script for Kaggle
# Use this to generate text samples from your trained model

# Set the subset size to match training
export OWT_SUBSET_NUM=11000

# Path to your trained checkpoint
# Update this to point to your actual checkpoint file
CHECKPOINT_PATH="${1:-lightning_logs/version_0/checkpoints/last.ckpt}"

echo "Running text generation on Kaggle"
echo "Using checkpoint: ${CHECKPOINT_PATH}"

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  algo.backbone=hf_dit \
  sampling.steps=8 \
  sampling.num_sample_batches=2 \
  sampling.noise_removal=greedy \
  loader.batch_size=1 \
  loader.eval_batch_size=1 \
  model.length=512 \
  eval.checkpoint_path="${CHECKPOINT_PATH}" \
  trainer.devices=1 \
  trainer.accelerator="gpu" \
  trainer.precision=16 \
  +wandb.offline=true

echo "Sampling complete!"
echo "Generated samples should be displayed above or saved to logs"
