import os

# 在导入 torch 之前设置环境变量以消除警告
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

import time
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator
from pathlib import Path

# ---------------- 基本配置 ----------------
MODEL_PATH = "./models/gpt-oss-20b"
# DATA_PATH = "./dataset/dataset_preprocessed_reduced.jsonl"
DATA_PATH = "./dataset/test.jsonl"
OUTPUT_DIR = "./output_dpo_pure"
SEED = 42

# ---------------- 初始化 ----------------
# 获取本地 rank 并在任何 CUDA 操作前设置当前设备
if "SLURM_LOCALID" in os.environ:
    # SLURM 提供的每个节点内本地 rank
    local_rank = int(os.environ["SLURM_LOCALID"])
elif "LOCAL_RANK" in os.environ:
    # accelerate 或 torchrun 方式启动
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    local_rank = 0  # fallback 单卡

torch.cuda.set_device(local_rank)
print(f"[Init] rank local={local_rank}, cuda device -> {torch.cuda.current_device()}")

set_seed(SEED)

# 初始化 Accelerator
accelerator = Accelerator(device_placement=True)
rank = accelerator.process_index
device = accelerator.device
is_main = accelerator.is_main_process
torch.set_float32_matmul_precision("high")

print(f"[Rank {os.environ.get('RANK', '?')}] uses cuda:{torch.cuda.current_device()}")

# ---------------- 工具函数 ----------------
def barrier_safe(timeout=120):
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()], timeout=torch.distributed.timedelta(seconds=timeout))
        except Exception as e:
            print(f"[Rank {rank}] barrier timeout: {e}")

def log(msg):
    if is_main:
        print(f"[Main] {msg}")

# ---------------- 加载数据 ----------------
def get_dataset(path):
    ds = load_dataset("json", data_files=path, split="train")
    return ds

# ---------------- 构建模型 ----------------
def get_lora_model():
    # 在 CPU 上加载模型，让 Accelerator 处理设备放置
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    return model

# ---------------- 构建 DPO 配置 ----------------
def get_dpo_args(tokenizer):
    return DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="no",  # 手动保存
        max_length=1024,
        beta=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        padding_value=tokenizer.pad_token_id,
        ddp_find_unused_parameters=False,
    )

# ---------------- 主训练流程 ----------------
def main():
    log("初始化中...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(DATA_PATH)
    model = get_lora_model()
    # 在 CPU 上加载 ref_model，让 Trainer 处理设备放置
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    dpo_args = get_dpo_args(tokenizer)
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
    )

    log("开始训练")
    start = time.time()
    trainer.train()
    duration = time.time() - start
    log(f"训练完成，用时 {duration:.2f}s")

    # ---------------- 安全保存 ----------------
    barrier_safe()
    trainer.accelerator.wait_for_everyone()
    log("保存 LoRA 适配器中...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 让所有 rank 都参与 state_dict 的构建（FSDP 需要全员参与）
    target = trainer.accelerator.unwrap_model(trainer.model)
    full_state = trainer.accelerator.get_state_dict(target)
    adapter_state = get_peft_model_state_dict(target, state_dict=full_state)
    # 仅在主进程写文件；其他 rank 跳过写入
    trainer.accelerator.save(adapter_state, os.path.join(OUTPUT_DIR, "adapter_model.bin"))
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(OUTPUT_DIR)
        log("模型保存完成")

    barrier_safe()
    torch.cuda.empty_cache()
    log("训练流程结束。")

# ---------------- 启动 ----------------
if __name__ == "__main__":
    main()
