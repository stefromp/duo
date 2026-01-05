"""
DUO Training on Kaggle - Python Script Version

This script provides a complete workflow for training DUO on Kaggle.
Upload this to a Kaggle notebook and run it with GPU enabled.

Usage in Kaggle:
    1. Create new notebook with GPU
    2. Upload this script
    3. Run: !python kaggle_train.py

Or run cells individually for more control.
"""

import os
import sys
import subprocess
import glob

print("=" * 70)
print("DUO Training on Kaggle")
print("=" * 70)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

SUBSET_SIZE = 11000  # Total documents (90% train, 10% val)
BATCH_SIZE = 2
SEQUENCE_LENGTH = 512
MAX_STEPS = 500
PRECISION = 16

print("Configuration:")
print(f"  Subset size: {SUBSET_SIZE} documents")
print(f"  Expected train: ~{int(SUBSET_SIZE * 0.9)} examples")
print(f"  Expected val: ~{int(SUBSET_SIZE * 0.1)} examples")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQUENCE_LENGTH}")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Precision: FP{PRECISION}")
print()

# ============================================================================
# STEP 1: CHECK GPU
# ============================================================================

print("Step 1: Checking GPU availability...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ GPU detected")
        # Print first few lines
        lines = result.stdout.split('\n')[:5]
        for line in lines:
            if line.strip():
                print(f"  {line}")
    else:
        print("✗ GPU not detected. Please enable GPU in Kaggle settings.")
        sys.exit(1)
except FileNotFoundError:
    print("✗ nvidia-smi not found. GPU may not be available.")
    sys.exit(1)

print()

# ============================================================================
# STEP 2: SETUP REPOSITORY
# ============================================================================

print("Step 2: Setting up repository...")
print()
print("⚠️  IMPORTANT: This script expects the modified DUO code to be uploaded")
print("    to your Kaggle notebook as a Dataset or directly in /kaggle/input/")
print()
print("    The code needs the following modifications:")
print("    1. dataloader.py - OpenWebText subset support")
print("    2. models/dit.py - FlashAttention made optional")
print()
print("    If you haven't uploaded the modified code, please do one of:")
print("    a) Upload this entire directory as a Kaggle Dataset")
print("    b) Clone from your own fork with modifications")
print("    c) Apply the patches manually after cloning")
print()

# Check if we're already in the DUO directory
if os.path.exists('main.py') and os.path.exists('dataloader.py'):
    print("✓ Already in DUO directory")
    print(f"  Working directory: {os.getcwd()}")
else:
    # Try to find DUO in Kaggle input
    if os.path.exists('/kaggle/input/duo'):
        print("✓ Found DUO in /kaggle/input/duo")
        print("  Copying to working directory...")
        import shutil
        shutil.copytree('/kaggle/input/duo', '/kaggle/working/duo')
        os.chdir('/kaggle/working/duo')
        print(f"✓ Working directory: {os.getcwd()}")
    else:
        # Clone from original repo (user will need to apply patches)
        print("⚠️  Cloning original repository (requires manual patching)")
        os.chdir('/kaggle/working')
        if not os.path.exists('duo'):
            result = subprocess.run(
                ['git', 'clone', 'https://github.com/kuleshov-group/discrete-diffusion-guidance.git', 'duo'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✓ Repository cloned")
                print()
                print("⚠️  IMPORTANT: You need to apply the modifications:")
                print("   1. Update dataloader.py for subset support")
                print("   2. Update models/dit.py to make FlashAttention optional")
                print()
                print("   See KAGGLE_GUIDE.md for details")
                print()
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting. Please apply modifications and run again.")
                    sys.exit(0)
            else:
                print("✗ Failed to clone repository")
                print(result.stderr)
                sys.exit(1)
        
        os.chdir('duo')
        print(f"✓ Working directory: {os.getcwd()}")

print()

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================

print("Step 3: Installing dependencies...")
print("(This may take 5-10 minutes)")

packages = [
    'torch', 'torchvision', 'torchaudio',
    'transformers', 'datasets', 'tokenizers',
    'pytorch-lightning',
    'hydra-core', 'omegaconf',
    'wandb', 'fsspec', 'aiohttp',
    'scipy', 'numpy', 'matplotlib'
]

print(f"Installing {len(packages)} packages...")
result = subprocess.run(
    ['pip', 'install', '-q'] + packages,
    capture_output=True, text=True
)

if result.returncode == 0:
    print("✓ All packages installed")
else:
    print("⚠ Some packages may have failed")
    if result.stderr:
        print(result.stderr[:500])  # Print first 500 chars of error

print()

# ============================================================================
# STEP 4: VERIFY SETUP
# ============================================================================

print("Step 4: Verifying setup...")

# Check if dataloader has subset support
try:
    with open('dataloader.py', 'r') as f:
        content = f.read()
    
    if 'OWT_SUBSET_NUM' in content:
        print("✓ Dataloader supports subset loading")
    else:
        print("⚠ Dataloader may need modification for subset loading")
        print("  The training will still work but will use full dataset")
except FileNotFoundError:
    print("✗ dataloader.py not found")
    sys.exit(1)

# Check main files
required_files = ['main.py', 'dataloader.py', 'configs/config.yaml']
all_exist = True
for filepath in required_files:
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
    else:
        print(f"✗ {filepath} not found")
        all_exist = False

if not all_exist:
    print("✗ Required files missing")
    sys.exit(1)

print()

# ============================================================================
# STEP 5: SET ENVIRONMENT VARIABLES
# ============================================================================

print("Step 5: Configuring environment...")
os.environ['OWT_SUBSET_NUM'] = str(SUBSET_SIZE)
os.environ['WANDB_MODE'] = 'offline'

print(f"✓ OWT_SUBSET_NUM={os.environ['OWT_SUBSET_NUM']}")
print(f"✓ WANDB_MODE={os.environ['WANDB_MODE']}")
print()

# ============================================================================
# STEP 6: RUN TRAINING
# ============================================================================

print("=" * 70)
print("Step 6: Starting training...")
print("=" * 70)
print()

training_cmd = [
    'python', 'main.py',
    'mode=train',
    'data=openwebtext-split',
    'algo=duo_base',
    'algo.backbone=hf_dit',
    f'loader.batch_size={BATCH_SIZE}',
    f'loader.global_batch_size={BATCH_SIZE}',
    f'loader.eval_batch_size={BATCH_SIZE}',
    f'model.length={SEQUENCE_LENGTH}',
    f'trainer.max_steps={MAX_STEPS}',
    'trainer.devices=1',
    'trainer.accelerator=gpu',
    f'trainer.precision={PRECISION}',
    'trainer.log_every_n_steps=50',
    'trainer.val_check_interval=0.2',
    '+wandb.offline=true'
]

print("Command:")
print(' \\\n  '.join(training_cmd))
print()

# Run training
result = subprocess.run(training_cmd)

if result.returncode != 0:
    print()
    print("✗ Training failed")
    sys.exit(1)

print()
print("✓ Training completed!")
print()

# ============================================================================
# STEP 7: CHECK RESULTS
# ============================================================================

print("=" * 70)
print("Step 7: Checking results...")
print("=" * 70)
print()

# Find checkpoints
checkpoints = glob.glob('lightning_logs/*/checkpoints/*.ckpt')
if checkpoints:
    print(f"✓ Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints:
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        print(f"  {ckpt} ({size_mb:.1f} MB)")
    
    latest_checkpoint = sorted(checkpoints)[-1]
    print()
    print(f"Latest checkpoint: {latest_checkpoint}")
else:
    print("⚠ No checkpoints found")

print()

# Find metrics
metrics_files = glob.glob('lightning_logs/*/metrics.csv')
if metrics_files:
    print(f"✓ Found metrics file: {metrics_files[0]}")
    try:
        import pandas as pd
        metrics = pd.read_csv(metrics_files[0])
        print()
        print("Last 5 training steps:")
        print(metrics.tail(5).to_string())
    except ImportError:
        print("  (Install pandas to view metrics: pip install pandas)")
else:
    print("⚠ No metrics file found")

print()

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("=" * 70)
print("Step 8: Saving results...")
print("=" * 70)
print()

output_dir = '/kaggle/working/duo_results'
os.makedirs(output_dir, exist_ok=True)

# Copy checkpoint
if checkpoints:
    import shutil
    output_checkpoint = os.path.join(output_dir, 'final_model.ckpt')
    shutil.copy(latest_checkpoint, output_checkpoint)
    print(f"✓ Checkpoint saved to: {output_checkpoint}")

# Copy metrics
if metrics_files:
    shutil.copy(metrics_files[0], os.path.join(output_dir, 'metrics.csv'))
    print(f"✓ Metrics saved to: {output_dir}/metrics.csv")

# Save configuration
import json
config = {
    'subset_size': SUBSET_SIZE,
    'batch_size': BATCH_SIZE,
    'sequence_length': SEQUENCE_LENGTH,
    'max_steps': MAX_STEPS,
    'precision': PRECISION,
    'checkpoint': latest_checkpoint if checkpoints else None
}

with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Configuration saved to: {output_dir}/config.json")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print()
print("Summary:")
print(f"  Total steps: {MAX_STEPS}")
print(f"  Dataset: OpenWebText subset ({SUBSET_SIZE} documents)")
print(f"  Results saved to: {output_dir}")
print()
print("Next steps:")
print("  1. Evaluate: python main.py mode=ppl_eval ...")
print("  2. Generate: python main.py mode=sample_eval ...")
print("  3. Download checkpoint from /kaggle/working/duo_results/")
print()
print("For more information, see:")
print("  - KAGGLE_GUIDE.md (comprehensive guide)")
print("  - KAGGLE_QUICKREF.md (quick reference)")
print("  - IMPLEMENTATION_SUMMARY.md (technical details)")
print()
print("=" * 70)
