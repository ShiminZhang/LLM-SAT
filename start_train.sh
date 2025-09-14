# --nodes=1               # 在1个节点内执行
# --ntasks-per-node=4     # 在这个节点上启动4个任务（也就是4个Python进程）
# --gpus-per-task=1       # 【关键】为每个任务分配1个独占的GPU
# --cpus-per-task=8       # (可选但推荐) 为每个GPU进程分配8个CPU核心，防止CPU瓶颈
# --mem-per-gpu=32G       # (可选但推荐) 为每个GPU进程分配64GB系统内存(RAM)

srun --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 python src/dpo.py