import json
import subprocess
import os

# 配置目标服务器信息
REMOTE_USER = "caixiao"
REMOTE_HOST = "121.48.161.104"
REMOTE_BASE_DIR = "/path/to/data/pv-view"  # 目标服务器上的绝对路径

# 读取 combined_path.json
with open("combined_path.json", "r") as f:
    folder_list = json.load(f)

new_paths = []

for folder in folder_list:
    folder_name = os.path.basename(folder.rstrip("/"))
    remote_path = f"{REMOTE_BASE_DIR}/{folder_name}"
    # 使用 scp 发送文件夹
    scp_cmd = [
        "scp", "-r", folder,
        f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}"
    ]
    subprocess.run(scp_cmd, check=True)
    # 记录新路径
    new_paths.append(remote_path)

# 生成新的 JSON 文件
with open("new_paths.json", "w") as f:
    json.dump(new_paths, f, indent=2)