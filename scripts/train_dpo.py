#!/usr/bin/env python3
"""
DPO Training Script for SAT Solver Code Generation

Fine-tunes Qwen 3-8B using Direct Preference Optimization (DPO)
to generate better SAT solver restart heuristic implementations.

Usage:
    python scripts/train_dpo.py --train_data data/dpo_formatted --output_dir outputs/dpo1/dpo_training
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_from_disk
import argparse

# Default model - Qwen 2.5-7B-Instruct
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def setup_model_and_tokenizer(model_name, use_qlora=True):
    """Load model with QLoRA configuration for memory-efficient training."""

    print(f"Loading model: {model_name}")
    print(f"Using QLoRA: {use_qlora}")

    # QLoRA configuration for 4-bit quantization
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    # Set pad token if not set (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for DPO

    # Load model
    # For standard training (no QLoRA), avoid quantization and let HF pick dtype
    if use_qlora:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )

    # Prepare for k-bit training
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    print(f"Model loaded. Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    return model, tokenizer


def setup_lora_config():
    """Configure LoRA parameters."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def main(args):
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DPO Training for SAT Solver Code Generation")
    print("=" * 60)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_qlora=args.use_qlora
    )

    # Load dataset
    print("\nLoading dataset...")
    train_dataset = load_from_disk(args.train_data)

    if args.eval_data:
        eval_dataset = load_from_disk(args.eval_data)
    else:
        # Split training data if no eval provided
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']

    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    # LoRA configuration (optional)
    peft_config = setup_lora_config() if args.use_lora else None

    # DPO Training configuration using DPOConfig
    training_args = DPOConfig(
        output_dir=args.output_dir,

        # Training hyperparameters
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,

        # DPO specific
        beta=args.beta,  # DPO temperature parameter

        # Optimization
        optim="paged_adamw_32bit",  # Memory efficient optimizer
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # Logging and saving
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,

        # Mixed precision
        bf16=True,

        # Other
        remove_unused_columns=False,
        report_to="none",  # No external logging (no WandB)
        run_name=args.run_name,

        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
    )

    # Initialize DPO Trainer
    print("\nInitializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting training...")
    print(f"Output directory: {args.output_dir}")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPO training for SAT solver code generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train_dpo.py --train_data data/dpo_formatted --output_dir outputs/dpo1/dpo_training

  # Custom hyperparameters
  python scripts/train_dpo.py --train_data data/dpo_formatted --output_dir outputs/dpo1/dpo_training \\
      --batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-5

  # Use different model
  python scripts/train_dpo.py --train_data data/dpo_formatted --model_name mistralai/Mistral-7B-Instruct-v0.3
        """
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help=f"Base model to fine-tune (default: {DEFAULT_MODEL})")
    parser.add_argument("--use_qlora", action="store_true", default=False,
                        help="Use QLoRA for memory efficiency (default: False)")
    parser.add_argument("--no_qlora", action="store_false", dest="use_qlora",
                        help="Disable QLoRA (requires more VRAM)")

    # LoRA on/off
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA adapters (default: True)")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora",
                        help="Disable LoRA (full fine-tuning; requires much more VRAM)")

    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training dataset (HuggingFace Dataset format)")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to evaluation dataset (optional, will split train if not provided)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/dpo1/dpo_training",
                        help="Output directory for model and checkpoints")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device (default: 2)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (overrides epochs if > 0)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter (default: 0.1)")

    # Sequence length arguments
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length (default: 1024)")

    # Run name
    parser.add_argument("--run_name", type=str, default="sat-solver-dpo",
                        help="Run name for logging (default: sat-solver-dpo)")

    args = parser.parse_args()
    main(args)