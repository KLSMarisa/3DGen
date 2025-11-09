import os

# --- 配置 ---
# 1. 设置包含待重命名文件夹的 **父目录**
parent_directory = '/mnt/hdd3/linzhuohang/3DGen/ckptv4/checkpoints/last.ckpt'

# 2. 设置要查找和替换的文件夹后缀
suffix_to_find = '-v2.ckpt'
new_suffix = '-v1.ckpt'
# --- 配置结束 ---

print("--- 开始执行文件夹重命名 ---")
print(f"目标父目录: '{parent_directory}'")
print("----------------------------------------")

# 检查父目录是否存在
if not os.path.isdir(parent_directory):
    print(f"错误：父目录 '{parent_directory}' 不存在。请检查路径是否正确。")
else:
    folders_renamed_count = 0
    try:
        # 遍历父目录中的所有项目
        for item_name in os.listdir(parent_directory):
            # 构建项目的完整路径
            old_path = os.path.join(parent_directory, item_name)

            # **核心改动：检查该项目是否是一个文件夹，并且名称符合条件**
            if os.path.isdir(old_path) and item_name.endswith(suffix_to_find):
                
                # 创建新的文件夹名称和路径
                new_folder_name = item_name[:-len(suffix_to_find)] + new_suffix
                new_path = os.path.join(parent_directory, new_folder_name)
                
                try:
                    # 执行重命名
                    os.rename(old_path, new_path)
                    print(f"成功: '{item_name}' -> '{new_folder_name}'")
                    folders_renamed_count += 1
                except OSError as e:
                    print(f"错误：重命名 '{item_name}' 失败。原因: {e}")

        if folders_renamed_count == 0:
            print(f"未找到任何以 '{suffix_to_find}' 结尾的文件夹进行重命名。")

    except Exception as e:
        print(f"处理目录时发生错误: {e}")

print("----------------------------------------")
print(f"操作完成。总共重命名了 {folders_renamed_count} 个文件夹。")