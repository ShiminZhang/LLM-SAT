#!/usr/bin/env python3
"""
Quick import test that runs on CPU (no GPU needed).
Tests if all packages and imports work without loading models.

Usage:
    conda activate dpo-training
    python scripts/test_imports_only.py
"""
import sys

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("Testing DPO Training Imports (CPU only)")
    print("=" * 60)

    errors = []

    # Test basic packages
    print("\n1. Testing basic packages...")
    try:
        import torch
        print(f"   ✓ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import transformers
        print(f"   ✓ transformers {transformers.__version__}")
        from packaging import version
        if  version.parse(transformers.__version__) < version.parse("4.37.0"):
            print(f"   ⚠ WARNING: transformers {transformers.__version__} < 4.37.0")
            print(f"     Qwen 2.5-7B-Instruct requires >= 4.37.0")
    except ImportError as e:
        errors.append(f"transformers: {e}")

    try:
        import peft
        print(f"   ✓ peft {peft.__version__}")
    except ImportError as e:
        errors.append(f"peft: {e}")

    try:
        import trl
        print(f"   ✓ trl {trl.__version__}")
    except ImportError as e:
        errors.append(f"trl: {e}")

    try:
        import bitsandbytes
        print(f"   ✓ bitsandbytes {bitsandbytes.__version__}")
        print(f"     (CPU warnings are normal, GPU will work on compute node)")
    except ImportError as e:
        # Only error if it's a real import issue, not triton.ops (GPU-only)
        if "triton" not in str(e):
            errors.append(f"bitsandbytes: {e}")
        else:
            print(f"   ⚠ bitsandbytes (GPU features unavailable on CPU, this is expected)")

    try:
        import datasets
        print(f"   ✓ datasets {datasets.__version__}")
    except ImportError as e:
        errors.append(f"datasets: {e}")

    # Test specific imports from train_dpo.py
    print("\n2. Testing train_dpo.py imports...")
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        print("   ✓ transformers imports")
    except ImportError as e:
        errors.append(f"transformers classes: {e}")

    try:
        from peft import LoraConfig, prepare_model_for_kbit_training
        print("   ✓ peft imports")
    except ImportError as e:
        errors.append(f"peft classes: {e}")

    try:
        from trl import DPOTrainer
        print("   ✓ trl.DPOTrainer")
    except ImportError as e:
        errors.append(f"trl.DPOTrainer: {e}")

    try:
        from datasets import load_from_disk
        print("   ✓ datasets.load_from_disk")
    except ImportError as e:
        errors.append(f"datasets.load_from_disk: {e}")

    # Test dataset loading
    print("\n3. Testing dataset loading...")
    try:
        from datasets import load_from_disk
        dataset = load_from_disk("data/dpo_formatted")
        print(f"   ✓ Loaded dataset with {len(dataset)} examples")
        print(f"   ✓ Dataset columns: {dataset.column_names}")

        # Check required columns
        required = {"prompt", "chosen", "rejected"}
        if required.issubset(set(dataset.column_names)):
            print(f"   ✓ All required columns present")
        else:
            missing = required - set(dataset.column_names)
            errors.append(f"Missing dataset columns: {missing}")
    except Exception as e:
        errors.append(f"dataset loading: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ ERRORS FOUND:")
        for err in errors:
            print(f"   - {err}")
        print("=" * 60)
        return False
    else:
        print("✓ All import tests passed!")
        print("=" * 60)
        print("\nYou can now submit the training job:")
        print("  sbatch scripts/submit_dpo.sh")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)