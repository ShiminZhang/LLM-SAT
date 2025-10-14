import gc
import logging
import os
import time

import psutil
import torch
import wandb

local_rank = int(os.environ.get("LOCAL_RANK", -1))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    if local_rank in (-1, 0):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
logger.propagate = False


def log_memory_usage(stage_name, wandb_enabled=False):
    """Record memory telemetry for the current process."""
    gpu_info = get_gpu_memory_info()
    cpu_memory = psutil.virtual_memory()

    memory_info = {
        "gpu_total_gb": gpu_info["total"] if gpu_info else 0,
        "gpu_allocated_gb": gpu_info["allocated"] if gpu_info else 0,
        "gpu_reserved_gb": gpu_info["reserved"] if gpu_info else 0,
        "gpu_free_gb": gpu_info["free"] if gpu_info else 0,
        "gpu_utilization_pct": gpu_info["utilization"] if gpu_info else 0,
        "cpu_memory_used_pct": cpu_memory.percent,
        "cpu_memory_available_gb": cpu_memory.available / 1024**3,
        "stage": stage_name,
    }

    if local_rank in (-1, 0):
        if gpu_info:
            logger.info(
                "📊 %s - GPU: %.1f%% used, %.1fGB free",
                stage_name,
                gpu_info["utilization"],
                gpu_info["free"],
            )
        else:
            logger.info("📊 %s - CPU: %.1f%% used", stage_name, cpu_memory.percent)

    if wandb_enabled and local_rank in (-1, 0):
        wandb.log(memory_info)

    return memory_info


def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return None

    try:
        device_index = torch.cuda.current_device()
    except AssertionError:
        device_index = 0

    if local_rank >= 0 and torch.cuda.device_count() > local_rank:
        device_index = local_rank

    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / 1024**3
    allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device_index) / 1024**3
    free_gb = max(total_gb - reserved_gb, 0.0)
    utilization = (reserved_gb / total_gb) * 100 if total_gb else 0.0

    return {
        "total": total_gb,
        "allocated": allocated_gb,
        "reserved": reserved_gb,
        "free": free_gb,
        "utilization": utilization,
    }


def force_cleanup_gpu():
    gpu_info = get_gpu_memory_info()
    if local_rank in (-1, 0):
        if gpu_info:
            logger.info(
                "Current GPU usage: %.1f%% used, %.2fGB allocated, %.2fGB reserved, %.2fGB free",
                gpu_info["utilization"],
                gpu_info["allocated"],
                gpu_info["reserved"],
                gpu_info["free"],
            )
        else:
            logger.info("No GPU available.")

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
