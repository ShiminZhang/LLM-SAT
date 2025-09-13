# import os
# import subprocess

# from huggingface_hub import snapshot_download, login, whoami
# TOKEN= "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"
# login(token=TOKEN)
# print(whoami())
# # 下载gpt-oss-20b模型（假设模型名称为"Open-Sora/gpt-oss-20b"）
# # model_name = "Open-Sora/gpt-oss-20b"
# # local_dir = "./models/gpt-oss-20b"
# # if not os.path.exists(local_dir):
# #     os.makedirs(local_dir)
# # snapshot_download(repo_id=model_name, local_dir=local_dir, resume_download=True)

# print(f"模型已下载到: {local_dir}")
from verl import DPOTrainer, DPOConfig