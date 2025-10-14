#!/bin/bash

# 多卡训练启动脚本
# 使用 torchrun 或 accelerate launch 启动分布式训练

echo "🚀 启动多卡DPO训练..."

# 方法1: 使用 accelerate launch
echo "使用 accelerate launch 启动训练..."
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port 29500 \
    src/dpo.py

# 方法2: 使用 torchrun (如果 accelerate launch 有问题)
# echo "使用 torchrun 启动训练..."
# torchrun \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     src/dpo.py

echo "✅ 训练完成"
