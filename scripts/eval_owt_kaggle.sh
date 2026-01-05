#!/bin/bash
# Evaluation script for running perplexity evaluation on Kaggle
# Use this after training to evaluate your trained model

# Set the subset size to match training
export OWT_SUBSET_NUM=11000

# Path to your trained checkpoint
# Update this to point to your actual checkpoint file
CHECKPOINT_PATH="${1:-lightning_logs/version_0/checkpoints/last.ckpt}"

echo "Running perplexity evaluation on Kaggle"
echo "Using checkpoint: ${CHECKPOINT_PATH}"
echo "Validation subset size: ~$((OWT_SUBSET_NUM / 10)) examples"

python main.py \
  mode=ppl_eval \
  data=openwebtext-split \
  algo=duo_base \
  algo.backbone=hf_dit \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  model.length=512 \
  eval.checkpoint_path="${CHECKPOINT_PATH}" \
  trainer.devices=1 \
  trainer.accelerator="gpu" \
  trainer.precision=16 \
  +wandb.offline=true

echo "Evaluation complete!"
