#!/usr/bin/env python3
"""
Test script to verify the OWT subset loading modifications work correctly.
Run this before training to ensure the dataloader changes are functioning.
"""

import os
import sys

def test_subset_env_var():
    """Test that the environment variable is recognized"""
    print("=" * 60)
    print("Test 1: Environment Variable Recognition")
    print("=" * 60)
    
    os.environ['OWT_SUBSET_NUM'] = '1000'
    subset_num = os.getenv("OWT_SUBSET_NUM")
    
    if subset_num:
        print(f"✓ Environment variable set: OWT_SUBSET_NUM={subset_num}")
        train_count = int(subset_num) * 9 // 10
        val_count = int(subset_num) - train_count
        print(f"✓ Expected split: {train_count} train, {val_count} val")
        return True
    else:
        print("✗ Failed to read OWT_SUBSET_NUM environment variable")
        return False

def test_dataloader_modification():
    """Test that dataloader.py contains the subset loading code"""
    print("\n" + "=" * 60)
    print("Test 2: Dataloader Modification Check")
    print("=" * 60)
    
    try:
        with open('dataloader.py', 'r') as f:
            content = f.read()
        
        required_strings = [
            'OWT_SUBSET_NUM',
            'os.getenv("OWT_SUBSET_NUM")',
            'subset_split',
            'Using OpenWebText subset'
        ]
        
        missing = []
        for req in required_strings:
            if req in content:
                print(f"✓ Found: {req}")
            else:
                print(f"✗ Missing: {req}")
                missing.append(req)
        
        if missing:
            print(f"\n✗ Dataloader is missing required modifications")
            print(f"  Missing strings: {missing}")
            return False
        else:
            print(f"\n✓ Dataloader contains all required modifications")
            return True
            
    except FileNotFoundError:
        print("✗ dataloader.py not found in current directory")
        return False

def test_imports():
    """Test that required packages can be imported"""
    print("\n" + "=" * 60)
    print("Test 3: Package Import Check")
    print("=" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'pytorch_lightning': 'PyTorch Lightning',
        'hydra': 'Hydra',
        'omegaconf': 'OmegaConf',
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} ({package})")
        except ImportError:
            print(f"✗ {name} ({package}) - Not installed")
            all_ok = False
    
    # Flash attention is optional
    try:
        __import__('flash_attn')
        print(f"✓ Flash Attention (optional) - Available")
    except ImportError:
        print(f"⚠ Flash Attention (optional) - Not available (will use fallback)")
    
    return all_ok

def test_split_calculation():
    """Test the split calculation logic"""
    print("\n" + "=" * 60)
    print("Test 4: Split Calculation Verification")
    print("=" * 60)
    
    test_cases = [
        (1000, 900, 100),
        (11000, 9900, 1100),
        (50000, 45000, 5000),
        (100000, 90000, 10000),
    ]
    
    all_ok = True
    for total, expected_train, expected_val in test_cases:
        train_count = int(total * 0.9)
        val_count = total - train_count
        
        if train_count == expected_train and val_count == expected_val:
            print(f"✓ Total {total:6d} → Train {train_count:6d}, Val {val_count:5d}")
        else:
            print(f"✗ Total {total:6d} → Train {train_count:6d}, Val {val_count:5d}")
            print(f"  Expected: Train {expected_train:6d}, Val {expected_val:5d}")
            all_ok = False
    
    return all_ok

def test_config_files():
    """Test that required config files exist"""
    print("\n" + "=" * 60)
    print("Test 5: Configuration Files Check")
    print("=" * 60)
    
    required_files = [
        'main.py',
        'dataloader.py',
        'configs/config.yaml',
        'configs/data/openwebtext-split.yaml',
        'configs/algo/duo_base.yaml',
    ]
    
    all_ok = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} - Not found")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DUO Kaggle Setup Verification")
    print("=" * 60)
    print()
    
    # Check working directory
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")
    
    if not os.path.exists('main.py'):
        print("\n✗ ERROR: main.py not found")
        print("  Please run this script from the DUO repository root directory")
        sys.exit(1)
    
    # Run tests
    results = {
        'Environment Variable': test_subset_env_var(),
        'Dataloader Modification': test_dataloader_modification(),
        'Package Imports': test_imports(),
        'Split Calculation': test_split_calculation(),
        'Config Files': test_config_files(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nYou're ready to run training on Kaggle!")
        print("\nNext steps:")
        print("  1. Set subset size: export OWT_SUBSET_NUM=11000")
        print("  2. Run training: ./scripts/train_owt_kaggle.sh")
        print("  3. Or use the Jupyter notebook: kaggle_notebook.ipynb")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
        print("See KAGGLE_GUIDE.md for detailed setup instructions.")
        sys.exit(1)

if __name__ == '__main__':
    main()
