import torch    
import gc
import time
import psutil
import logging
import wandb

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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