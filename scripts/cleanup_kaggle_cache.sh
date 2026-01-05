#!/bin/bash
# Cleanup script to free up disk space on Kaggle before training

echo "Cleaning up Kaggle disk space..."

# Remove default HuggingFace cache that may have accumulated
if [ -d "/kaggle/working/hf" ]; then
    echo "Removing /kaggle/working/hf cache..."
    rm -rf /kaggle/working/hf
    echo "Removed /kaggle/working/hf"
fi

# Remove old checkpoints if they exist
if [ -d "/kaggle/working/duo/duo/duo/duo/outputs" ]; then
    echo "Removing old training outputs..."
    rm -rf /kaggle/working/duo/duo/duo/duo/outputs
    echo "Removed old outputs"
fi

# Show disk usage
echo ""
echo "Current disk usage:"
df -h /kaggle/working

echo ""
echo "Cleanup complete!"
