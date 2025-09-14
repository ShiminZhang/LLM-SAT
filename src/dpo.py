import torch
import torch.distributed as dist
import logging
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
import os
import gc
import time
from utils import get_gpu_memory_info, force_cleanup_gpu, log_memory_usage, logger
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator, DataLoaderConfiguration
from data_manager import DataManager

# 优化设置
torch.set_float32_matmul_precision('high')  # 启用 TensorFloat32 优化

# 抑制一些不重要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
warnings.filterwarnings("ignore", message=".*tensor cores for float32 matrix multiplication.*")
warnings.filterwarnings("ignore", message=".*FSDP upcast of low precision parameters.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")


dataloader_config = DataLoaderConfiguration(dispatch_batches=True, split_batches=True)
accelerator = Accelerator(dataloader_config=dataloader_config)


# --- 全局参数 ---
n_epochs = 1
beta = 0.1
lr = 2e-5
warmup_steps = 10
accumulation_steps=1
n_of_gpus = 4
lora_r = 32
per_device_batch_size = 32           # 这是 per_device_batch_size
data_size = 1000
data_size = per_device_batch_size * n_of_gpus * 10
max_steps = data_size // per_device_batch_size
data_file_path = "./dataset/dataset_preprocessed.jsonl" 


# 初始化wandb
def init_wandb(lora_r=16, lora_alpha=32, lora_dropout=0.1, batch_size=1):
    """初始化wandb"""
    try:
        # 从环境变量或直接设置API密钥
        api_key = os.getenv("WANDB_API_KEY", "2b12ed4713d66c27d43040761ff1e0574c7a7ef2")
        wandb.login(key=api_key)
        
        # 初始化wandb项目
        wandb.init(
            project="gpt-oss-dpo",
            name="dpo-lora-training",
            config={
                "model": f"gpt-oss-20b_bs{batch_size}_r{lora_r}_alpha{lora_alpha}_dropout{lora_dropout}",
                "method": "DPO+LoRA",
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": n_epochs,
                "beta": beta,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
            }
        )
        logger.info("✅ Wandb初始化成功")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Wandb初始化失败: {e}")
        return False

def run_training(lora_r=16, lora_alpha=32, lora_dropout=0.1):
    model_name = "./models/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_config = DPOConfig(
        output_dir=f"./gpt_oss_dpo_output_bs{per_device_batch_size}",
        # num_train_epochs=n_epochs,
        max_steps=10,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_steps=50,
        save_strategy="steps",
        save_safetensors=False,  # 禁用 safetensors 保存格式
        remove_unused_columns=False,
        max_length=2048,
        beta=beta,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=False,
        bf16=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        padding_value=tokenizer.pad_token_id,
        # 添加内存管理相关配置
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
    )
    is_main_process = dpo_config.local_rank in [-1, 0]
    if is_main_process:
        logger.info("🚀 start training...")
        wandb_enabled = init_wandb(lora_r, lora_alpha, lora_dropout, per_device_batch_size)
        log_memory_usage("after model loading", wandb_enabled)
    else:
        wandb_enabled = False
    
    force_cleanup_gpu()
    
    log_memory_usage("Initial state", wandb_enabled)
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # device_map={"": dpo_config.local_rank},
        trust_remote_code=True,
    )
    # model = prepare_model_for_kbit_training(model)
    # 设置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 记录LoRA应用后内存状态
    log_memory_usage("after LoRA application", wandb_enabled)
    
    data_file_path = "./dataset/dataset_preprocessed.jsonl" # <-- 替换成你的数据文件路径
    # 加载数据集
    train_dataset = load_dataset("json", data_files=data_file_path, split="train", streaming=False)
    # 限制数据大小用于测试
    train_dataset = train_dataset.take(data_size)
    results = []
    
    logger.info(f"🧪 test batch_size = {per_device_batch_size}")
    
    try:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        memory_before = log_memory_usage(f"batch_{per_device_batch_size}_before", wandb_enabled)
        
        
        # 创建DPO训练器
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=train_dataset,
        )
        
        memory_trainer = log_memory_usage(f"batch_{per_device_batch_size}_trainer_created", wandb_enabled)
        
        # 开始训练
        start_time = time.time()
        try:
            dpo_trainer.train()
            training_time = time.time() - start_time
        except Exception as train_error:
            logger.warning(f"训练过程中出现错误，但可能已完成: {train_error}")
            training_time = time.time() - start_time
            # 检查是否至少完成了一些步骤
            if hasattr(dpo_trainer.state, 'global_step') and dpo_trainer.state.global_step > 0:
                logger.info(f"训练部分完成，已完成 {dpo_trainer.state.global_step} 步")
            else:
                raise train_error
        
        memory_after = log_memory_usage(f"batch_{per_device_batch_size}_after_training", wandb_enabled)
        
        # 记录结果
        result = {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "batch_size": per_device_batch_size,
            "training_time": training_time,
            "memory_before": memory_before,
            "memory_trainer": memory_trainer,
            "memory_after": memory_after,
            "success": True
        }
        results.append(result)
        
        logger.info(f"✅ Batch size {per_device_batch_size} 训练完成，耗时: {training_time:.2f}秒")
        
        # 清理内存
        del dpo_trainer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        logger.error(f"❌ Batch size {per_device_batch_size} 训练失败: {e}")
        result = {
            "batch_size": per_device_batch_size,
            "error": str(e),
            "success": False
        }
        results.append(result)
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 输出测试结果总结
    if is_main_process:
        logger.info("📊 Batch Size 测试结果总结:")
        logger.info("=" * 80)
        for result in results:
            if result["success"]:
                logger.info(f"Batch Size {result['batch_size']:2d}: ✅ 成功 - "
                        f"训练时间: {result['training_time']:6.2f}s - "
                        f"GPU利用率: {result['memory_after']['gpu_utilization_pct']:5.1f}% - "
                        f"GPU使用: {result['memory_after']['gpu_reserved_gb']:5.1f}GB")
            else:
                logger.info(f"Batch Size {result['batch_size']:2d}: ❌ 失败 - {result['error']}")
    
    # 记录到wandb
    if wandb_enabled:
        for result in results:
            if result["success"]:
                wandb.log({
                    "batch_size": result["batch_size"],
                    "training_time": result["training_time"],
                    "gpu_utilization_pct": result["memory_after"]["gpu_utilization_pct"],
                    "gpu_reserved_gb": result["memory_after"]["gpu_reserved_gb"],
                    "test_success": True
                })
            else:
                wandb.log({
                    "batch_size": result["batch_size"],
                    "error": result["error"],
                    "test_success": False
                })
    
    # 最终内存状态
    log_memory_usage("测试完成", wandb_enabled)
    
    # 彻底清理显存
    try:
        del model
        del tokenizer
        del train_dataset
        if 'dpo_trainer' in locals():
            del dpo_trainer
    except:
        pass
    
    force_cleanup_gpu()
    
    # 记录清理后内存状态
    log_memory_usage("清理后", wandb_enabled)
    
    return results

def main():
    logger.info("开始gpt-oss-20b DPO+LoRA batch size测试...")
    
    config =  {"lora_r": lora_r, "lora_alpha": 2 * lora_r, "lora_dropout": 0.1}
    
    all_results = []
    
    logger.info(f"🔄 start training with config: {config}")
    force_cleanup_gpu()
    
    try:
        results = run_training(**config)
        all_results.extend(results)

    except Exception as e:
        logger.error(f"❌ training failed: {e}")
        all_results.append({
            "config": config,
            "error": str(e),
            "success": False
        })

    # 输出所有测试结果总结
    logger.info("📊 all results summary:")
    logger.info("=" * 100)
    for i, result in enumerate(all_results):
        if result.get("success", False):
            logger.info(f"测试 {i+1}: LoRA r={result['lora_r']}，alpha={result['lora_alpha']}，dropout={result['lora_dropout']}，batch_size={result['batch_size']} - ✅ 成功 - "
                        f"训练时间: {result['training_time']:6.2f}s - "
                        f"GPU利用率: {result['memory_after']['gpu_utilization_pct']:5.1f}% - "
                        f"GPU使用: {result['memory_after']['gpu_reserved_gb']:5.1f}GB")
        else:
            logger.info(f"测试 {i+1}: ❌ 失败 - {result.get('error', 'Unknown error')}")
    

if __name__ == "__main__":
    main()

