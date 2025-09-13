#!/usr/bin/env python3
"""
使用gpt-oss-20b模型的DPO+LoRA测试脚本
"""
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import wandb
import os
import psutil
import time
import gc
from data_manager import DataManager

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
                "learning_rate": 5e-6,
                "batch_size": batch_size,
                "epochs": 1,
                "beta": 0.1,
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
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_memory_info():
    """获取GPU显存信息"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
        gpu_free = gpu_memory - gpu_reserved
        return {
            "total": gpu_memory,
            "allocated": gpu_allocated,
            "reserved": gpu_reserved,
            "free": gpu_free,
            "utilization": (gpu_reserved / gpu_memory) * 100
        }
    return None

def force_cleanup_gpu():
    """强制清理GPU显存"""
    if torch.cuda.is_available():
        # 清理PyTorch缓存
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # 清理Python垃圾回收
        gc.collect()
        
        # 强制同步，确保清理完成
        torch.cuda.synchronize()
        
        # 再次清理
        torch.cuda.empty_cache()
        gc.collect()
        
        # 等待一小段时间让系统处理
        time.sleep(1)

def log_memory_usage(stage_name, wandb_enabled=False):
    """记录内存使用情况"""
    gpu_info = get_gpu_memory_info()
    cpu_memory = psutil.virtual_memory()
    
    memory_info = {
        f"gpu_total_gb": gpu_info["total"] if gpu_info else 0,
        f"gpu_allocated_gb": gpu_info["allocated"] if gpu_info else 0,
        f"gpu_reserved_gb": gpu_info["reserved"] if gpu_info else 0,
        f"gpu_free_gb": gpu_info["free"] if gpu_info else 0,
        f"gpu_utilization_pct": gpu_info["utilization"] if gpu_info else 0,
        f"cpu_memory_used_pct": cpu_memory.percent,
        f"cpu_memory_available_gb": cpu_memory.available / 1024**3,
        "stage": stage_name
    }
    
    logger.info(f"📊 {stage_name} - GPU: {gpu_info['utilization']:.1f}% used, {gpu_info['free']:.1f}GB free" if gpu_info else f"📊 {stage_name} - CPU: {cpu_memory.percent:.1f}% used")
    
    if wandb_enabled:
        wandb.log(memory_info)
    
    return memory_info

    
    # 为了测试不同batch size，我们重复数据
    extended_data = []
    for i in range(5):  # 重复5次，总共75个样本
        for item in base_data:
            extended_data.append(item)
    
    return extended_data

def test_batch_sizes(lora_r=16, lora_alpha=32, lora_dropout=0.1, batch_size=1):
    """测试不同batch size对GPU显存的影响"""
    logger.info("🚀 开始batch size显存测试...")
    
    # 初始化wandb
    wandb_enabled = init_wandb(lora_r, lora_alpha, lora_dropout, batch_size)
    
    # 使用本地gpt-oss-20b模型
    model_name = "./models/gpt-oss-20b"
    logger.info(f"加载模型: {model_name}")
    
    # 强制清理显存
    force_cleanup_gpu()
    
    # 记录初始内存状态
    log_memory_usage("初始状态", wandb_enabled)
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 记录模型加载后内存状态
    log_memory_usage("模型加载后", wandb_enabled)
    
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
    log_memory_usage("LoRA应用后", wandb_enabled)
    
    # 准备数据
    raw_data = DataManager().get_data(1000)
    train_dataset = Dataset.from_list(raw_data)
    logger.info(f"📊 数据集大小: {len(train_dataset)} 个样本")
    # input("按回车键继续测试不同batch size...")  # wait for enter to continue
    # 测试不同的batch size
    results = []
    
    logger.info(f"🧪 测试 batch_size = {batch_size}")
    
    try:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 记录测试前内存状态
        memory_before = log_memory_usage(f"batch_{batch_size}_before", wandb_enabled)
        
        # DPO配置
        dpo_config = DPOConfig(
            output_dir=f"./gpt_oss_dpo_output_bs{batch_size}",
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=2,
            logging_steps=1,
            save_steps=50,
            save_strategy="steps",
            remove_unused_columns=False,
            max_length=512,
            beta=0.1,
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            gradient_accumulation_steps=1,
            bf16=True,
            dataloader_num_workers=0,
            padding_value=tokenizer.pad_token_id,
        )
        
        # 创建DPO训练器
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=train_dataset,
        )
        
        # 记录训练器创建后内存状态
        memory_trainer = log_memory_usage(f"batch_{batch_size}_trainer_created", wandb_enabled)
        
        # 开始训练
        start_time = time.time()
        dpo_trainer.train()
        training_time = time.time() - start_time
        
        # 记录训练后内存状态
        memory_after = log_memory_usage(f"batch_{batch_size}_after_training", wandb_enabled)
        
        # 记录结果
        result = {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "batch_size": batch_size,
            "training_time": training_time,
            "memory_before": memory_before,
            "memory_trainer": memory_trainer,
            "memory_after": memory_after,
            "success": True
        }
        results.append(result)
        
        logger.info(f"✅ Batch size {batch_size} 训练完成，耗时: {training_time:.2f}秒")
        
        # 清理内存
        del dpo_trainer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        logger.error(f"❌ Batch size {batch_size} 训练失败: {e}")
        result = {
            "batch_size": batch_size,
            "error": str(e),
            "success": False
        }
        results.append(result)
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 输出测试结果总结
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
    
    # 测试配置列表
    test_configs = [
        {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.1, "batch_size": 16},
        {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.1, "batch_size": 64},
        {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.1, "batch_size": 128},
        {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.1, "batch_size": 256},
        # {"lora_r": 48, "lora_alpha": 96, "lora_dropout": 0.1, "batch_size": 16},
        {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.1, "batch_size": 16},
        {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.1, "batch_size": 64},
        {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.1, "batch_size": 128},
        {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.1, "batch_size": 256},
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs):
        logger.info(f"🔄 开始第 {i+1}/{len(test_configs)} 次测试: LoRA r={config['lora_r']}")
        
        try:
            results = test_batch_sizes(**config)
            all_results.extend(results)
            
            # 在测试之间添加等待和清理
            if i < len(test_configs) - 1:  # 不是最后一次测试
                logger.info("⏳ 等待5秒让显存完全释放...")
                time.sleep(5)
                
                # 强制清理显存
                force_cleanup_gpu()
                
                # 记录清理后状态
                gpu_info = get_gpu_memory_info()
                if gpu_info:
                    logger.info(f"🧹 清理后显存状态: {gpu_info['utilization']:.1f}% used, {gpu_info['free']:.1f}GB free")
                
        except Exception as e:
            logger.error(f"❌ 第 {i+1} 次测试失败: {e}")
            all_results.append({
                "config": config,
                "error": str(e),
                "success": False
            })
    
    # 输出所有测试结果总结
    logger.info("📊 所有测试结果总结:")
    logger.info("=" * 100)
    for i, result in enumerate(all_results):
        if result.get("success", False):
            logger.info(f"测试 {i+1}: LoRA r={result['lora_r']}，alpha={result['lora_alpha']}，dropout={result['lora_dropout']}，batch_size={result['batch_size']} - ✅ 成功 - "
                       f"训练时间: {result['training_time']:6.2f}s - "
                       f"GPU利用率: {result['memory_after']['gpu_utilization_pct']:5.1f}% - "
                       f"GPU使用: {result['memory_after']['gpu_reserved_gb']:5.1f}GB")
        else:
            logger.info(f"测试 {i+1}: ❌ 失败 - {result.get('error', 'Unknown error')}")
    
    logger.info("🎉 所有测试完成！")

        

if __name__ == "__main__":
    main()

