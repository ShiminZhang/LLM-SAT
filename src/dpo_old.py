我的多卡训练脚本在保存的时候老是卡住，怎么办
import argparse
import gc
import logging
import os
from datetime import datetime
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

import wandb

from utils import force_cleanup_gpu, log_memory_usage, logger as base_logger

# 优化设置
torch.set_float32_matmul_precision('high')

# 抑制一些不重要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
warnings.filterwarnings("ignore", message=".*tensor cores for float32 matrix multiplication.*")
warnings.filterwarnings("ignore", message=".*FSDP upcast of low precision parameters.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")


logger = base_logger


@dataclass
class TrainingConfig:
    model_name: str = "./models/gpt-oss-20b"
    dataset_path: str = "./dataset/dataset_preprocessed_reduced.jsonl"
    output_dir: Optional[str] = None
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: Optional[int] = None
    max_steps: Optional[int] = 10
    warmup_steps: int = 10
    logging_steps: int = 1
    save_steps: int = 50
    save_strategy: str = "steps"
    beta: float = 0.1
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    ddp_find_unused_parameters: bool = False
    bf16: bool = True
    max_length: int = 2048
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    dataloader_drop_last: bool = True
    lora_r: int = 32
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    seed: int = 42
    enable_wandb: bool = True
    wandb_project: str = "gpt-oss-dpo"
    wandb_run_name: Optional[str] = None
    wandb_api_key_env: str = "WANDB_API_KEY"
    limit_samples: Optional[int] = None
    streaming: bool = False
    resume_from_checkpoint: Optional[str] = None
    save_final_model: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-OSS DPO with LoRA using Accelerate/TRL.")
    parser.add_argument("--model-name", default=TrainingConfig.model_name)
    parser.add_argument("--dataset-path", default=TrainingConfig.dataset_path)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=TrainingConfig.per_device_train_batch_size)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=TrainingConfig.per_device_eval_batch_size)
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=TrainingConfig.warmup_steps)
    parser.add_argument("--logging-steps", type=int, default=TrainingConfig.logging_steps)
    parser.add_argument("--save-steps", type=int, default=TrainingConfig.save_steps)
    parser.add_argument("--save-strategy", default=TrainingConfig.save_strategy)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--lr-scheduler-type", default=TrainingConfig.lr_scheduler_type)
    parser.add_argument("--beta", type=float, default=TrainingConfig.beta)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=TrainingConfig.gradient_accumulation_steps)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--max-length", type=int, default=TrainingConfig.max_length)
    parser.add_argument("--dataloader-num-workers", type=int, default=TrainingConfig.dataloader_num_workers)
    parser.add_argument("--dataloader-pin-memory", action="store_true")
    parser.add_argument("--dataloader-persistent-workers", action="store_true")
    parser.add_argument("--no-drop-last", action="store_true")
    parser.add_argument("--lora-r", type=int, default=TrainingConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=TrainingConfig.lora_dropout)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=TrainingConfig.wandb_project)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-api-key-env", default=TrainingConfig.wandb_api_key_env)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--no-save-final-model", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    cfg = TrainingConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        beta=args.beta,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing and not args.no_gradient_checkpointing,
        bf16=args.bf16 or not args.no_bf16,
        max_length=args.max_length,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_drop_last=not args.no_drop_last,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        enable_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_api_key_env=args.wandb_api_key_env,
        limit_samples=args.limit_samples,
        streaming=args.streaming,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_final_model=not args.no_save_final_model,
    )

    if cfg.output_dir is None:
        date = datetime.now().strftime("%Y%m%d")
        cfg.output_dir = f"./gpt_oss_dpo_{cfg.per_device_train_batch_size}_{date}"

    if cfg.lora_alpha is None:
        cfg.lora_alpha = 2 * cfg.lora_r

    if cfg.wandb_run_name is None:
        cfg.wandb_run_name = Path(cfg.output_dir).name

    return cfg


def _config_for_wandb(config: TrainingConfig) -> Dict[str, object]:
    payload = asdict(config)
    payload["target_modules"] = list(config.target_modules)
    return payload


def setup_wandb_if_needed(config: TrainingConfig, is_main_process: bool) -> bool:
    if not config.enable_wandb or not is_main_process:
        return False

    try:
        api_key = os.getenv(config.wandb_api_key_env)
        if api_key:
            wandb.login(key=api_key)
    except Exception as exc:  # pragma: no cover
        logger.warning("⚠️ Wandb 登录失败: %s", exc)

    try:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=_config_for_wandb(config),
        )
        logger.info("✅ Wandb 初始化成功")
        return True
    except Exception as exc:
        logger.warning("⚠️ Wandb 初始化失败: %s", exc)
        return False


def load_tokenizer(config: TrainingConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_training_dataset(config: TrainingConfig):
    dataset = load_dataset("json", data_files=config.dataset_path, split="train", streaming=config.streaming)

    if config.limit_samples is None:
        return dataset

    if getattr(dataset, "take", None) is not None:
        return dataset.take(config.limit_samples)

    try:
        length = len(dataset)
    except TypeError:
        length = None

    if length is not None:
        limit = min(config.limit_samples, length)
        return dataset.select(range(limit))

    return dataset  # fallback


def build_model(config: TrainingConfig) -> AutoModelForCausalLM:
    model_kwargs = {"trust_remote_code": True}
    # if config.bf16:
    #     model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        # dtype="bfloat16",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_dpo_config(config: TrainingConfig, tokenizer: AutoTokenizer) -> DPOConfig:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy=config.save_strategy,
        remove_unused_columns=False,
        max_length=config.max_length,
        beta=config.beta,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_persistent_workers=config.dataloader_persistent_workers,
        dataloader_drop_last=config.dataloader_drop_last,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        padding_value=tokenizer.pad_token_id,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )

    if config.num_train_epochs is not None:
        dpo_kwargs["num_train_epochs"] = config.num_train_epochs

    if config.max_steps is not None:
        dpo_kwargs["max_steps"] = config.max_steps

    return DPOConfig(**dpo_kwargs)


def finalize_run(trainer: Optional[DPOTrainer], wandb_enabled: bool, is_main_process: bool) -> None:
    accelerator = getattr(trainer, "accelerator", None) if trainer is not None else None
    if accelerator is not None:
        try:
            accelerator.wait_for_everyone()
            accelerator.free_memory()
        except Exception as exc:  # pragma: no cover
            logger.warning("⚠️ Accelerator 清理失败: %s", exc)

    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except RuntimeError:
            pass

    if wandb_enabled and is_main_process:
        try:
            wandb.finish()
        except Exception as exc:  # pragma: no cover
            logger.warning("⚠️ Wandb 关闭失败: %s", exc)

    force_cleanup_gpu()
    torch.cuda.empty_cache()
    gc.collect()


def run_training(config: TrainingConfig) -> Dict[str, float]:
    trainer: Optional[DPOTrainer] = None
    wandb_enabled = False
    is_main_process = True

    try:
        set_seed(config.seed)

        tokenizer = load_tokenizer(config)
        train_dataset = load_training_dataset(config)
        model = build_model(config)
        dpo_args = build_dpo_config(config, tokenizer)
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            # dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_args,
            train_dataset=train_dataset,
        )

        accelerator = getattr(trainer, "accelerator", None)
        is_main_process = accelerator.is_main_process if accelerator is not None else dpo_args.local_rank in (-1, 0)

        wandb_enabled = setup_wandb_if_needed(config, is_main_process)

        log_memory_usage("initial_state", wandb_enabled)

        start_time = time.time()
        train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        duration = time.time() - start_time

        metrics = train_result.metrics or {}
        metrics["training_duration_sec"] = duration
        log_memory_usage("after_training", wandb_enabled)

        if wandb_enabled:
            numeric_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
            if numeric_metrics:
                wandb.log(numeric_metrics)

        # if config.save_final_model and is_main_process:
        #     # Only persist the LoRA adapter to avoid consolidating the massive FSDP full state dict.
        #     target_model = trainer.accelerator.unwrap_model(trainer.model) if accelerator is not None else trainer.model
        #     target_model.save_pretrained(config.output_dir)
        #     tokenizer.save_pretrained(config.output_dir)
        if accelerator.is_main_process:
            logger.info("🚀 Saving model...")
            target_model = trainer.accelerator.unwrap_model(trainer.model) if accelerator is not None else trainer.model
            target_model.save_pretrained(config.output_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(config.output_dir)
            logger.info("Saving done")

        accelerator.wait_for_everyone()


        if is_main_process:
            logger.info("✅ 训练完成 - 用时 %.2fs", duration)
            for key, value in metrics.items():
                logger.info("• %s: %s", key, value)

        return metrics
    finally:
        finalize_run(trainer, wandb_enabled, is_main_process)


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    logger.info("🚀 启动 DPO 训练，配置: %s", config)

    run_training(config)


if __name__ == "__main__":
    main()