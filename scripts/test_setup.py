#!/usr/bin/env python3
"""
Quick test to verify DPO training setup before submitting full training job.

Tests:
1. CUDA/GPU availability
2. Model loading with QLoRA
3. Basic inference
4. Required package versions

Usage:
    python scripts/test_setup.py
    python scripts/test_setup.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import sys

# Default model - same as train_dpo.py
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")
    errors = []

    try:
        import torch
        print(f"  torch: {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
        # Check version for Llama 3.1 support
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.43.0"):
            print(f"    WARNING: transformers {transformers.__version__} may not support Llama 3.1")
            print(f"    Recommended: >= 4.43.0")
    except ImportError as e:
        errors.append(f"transformers: {e}")

    try:
        import peft
        print(f"  peft: {peft.__version__}")
    except ImportError as e:
        errors.append(f"peft: {e}")

    try:
        import trl
        print(f"  trl: {trl.__version__}")
        # Check if DPOTrainer is available
        from trl import DPOTrainer
        print(f"    DPOTrainer: available")
    except ImportError as e:
        errors.append(f"trl: {e}")

    try:
        import bitsandbytes
        print(f"  bitsandbytes: {bitsandbytes.__version__}")
    except ImportError as e:
        errors.append(f"bitsandbytes: {e}")

    try:
        import datasets
        print(f"  datasets: {datasets.__version__}")
    except ImportError as e:
        errors.append(f"datasets: {e}")

    try:
        import accelerate
        print(f"  accelerate: {accelerate.__version__}")
    except ImportError as e:
        errors.append(f"accelerate: {e}")

    if errors:
        print("\n❌ Missing packages:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("✓ All required packages available")
    return True


def test_cuda():
    """Test CUDA availability and GPU info."""
    import torch

    print("\nTesting CUDA...")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
        print("✓ CUDA setup OK")
        return True
    else:
        print("❌ CUDA not available - GPU training will not work")
        return False


def test_model_loading(model_name):
    """Test model loading with QLoRA configuration."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"\nTesting model loading: {model_name}")
    print("  (This may take a few minutes on first run to download the model)")

    try:
        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load tokenizer
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"    Vocab size: {tokenizer.vocab_size}")

        # Load model
        print("  Loading model with QLoRA (4-bit)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        memory_gb = model.get_memory_footprint() / 1e9
        print(f"  Model memory footprint: {memory_gb:.2f} GB")

        print("✓ Model loaded successfully")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None


def test_inference(model, tokenizer):
    """Test basic inference."""
    print("\nTesting inference...")

    try:
        prompt = "bool kissat_restarting(kissat *solver) {"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print(f"  Prompt: {prompt}")
        print("  Generating...")

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated:\n{generated[:200]}...")
        print("✓ Inference works")
        return True

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False


def main(args):
    print("=" * 60)
    print("DPO Training Setup Verification")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\n❌ Setup verification FAILED - missing packages")
        sys.exit(1)

    # Test CUDA
    if not test_cuda():
        print("\n❌ Setup verification FAILED - CUDA not available")
        sys.exit(1)

    # Test model loading
    model, tokenizer = test_model_loading(args.model)
    if model is None:
        print("\n❌ Setup verification FAILED - model loading failed")
        print("\nIf you see an authentication error, run:")
        print("  huggingface-cli login")
        print("  https://huggingface.co/Qwen/Qwen3-8B")
        sys.exit(1)

    # Test inference
    if not test_inference(model, tokenizer):
        print("\n❌ Setup verification FAILED - inference failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ All tests passed! Setup verification complete.")
    print("=" * 60)
    print("\nYou can now run DPO training with:")
    print("  sbatch scripts/submit_dpo.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DPO training setup")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to test (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    main(args)