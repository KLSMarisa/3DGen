import os
import json
import matplotlib.pyplot as plt
import numpy as np

def process_and_plot_data(root_dir):
    """
    遍历指定目录下的文件夹，汇总JSON数据，计算平均值，并绘制折线图。

    Args:
        root_dir (str): 包含各大文件夹的根目录路径。
    """
    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：目录 '{root_dir}' 不存在。")
        return

    all_results = {}
    # 获取所有按数字顺序排序的大文件夹
    dir_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()], key=int)

    if not dir_names:
        print(f"错误：在 '{root_dir}' 中没有找到符合条件的数字文件夹。")
        return

    # 遍历每个大文件夹 (e.g., '12000', '12500')
    for dir_name in dir_names:
        main_folder_path = os.path.join(root_dir, dir_name)
        
        # 用于存储当前大文件夹下所有小文件夹数据的总和与计数
        aggregated_data = {
            "0": {"psnr": 0, "ssim": 0, "count": 0},
            "1": {"psnr": 0, "ssim": 0, "count": 0},
            "2": {"psnr": 0, "ssim": 0, "count": 0},
        }

        # 遍历小文件夹 (0 to 19)
        for i in range(21):
            sub_folder_path = os.path.join(main_folder_path, str(i))
            json_file_path = os.path.join(sub_folder_path, 'compare.json')

            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                        # 累加 psnr 和 ssim
                        for key in ["0", "1", "2"]:
                            if key in data:
                                aggregated_data[key]["psnr"] += data[key].get("psnr", 0)
                                aggregated_data[key]["ssim"] += data[key].get("ssim", 0)
                                aggregated_data[key]["count"] += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"警告：读取或解析文件 '{json_file_path}' 时出错: {e}")
            else:
                print(f"警告：文件 '{json_file_path}' 未找到，已跳过。")

        # 计算平均值
        avg_results = {}
        for key, values in aggregated_data.items():
            count = values["count"]
            if count > 0:
                avg_results[key] = {
                    "psnr": values["psnr"] / count,
                    "ssim": values["ssim"] / count
                }
            else:
                # 如果没有数据，则设为0或NaN
                avg_results[key] = {"psnr": 0, "ssim": 0}
        
        all_results[dir_name] = avg_results

    # --- 数据准备与绘图 ---
    if not all_results:
        print("没有处理任何数据，无法绘图。")
        return

    labels = sorted(all_results.keys(), key=int)
    
    # 准备绘图数据
    plot_data = {
        "psnr": {"0": [], "1": [], "2": [], "all": []},
        "ssim": {"0": [], "1": [], "2": [], "all": []}
    }

    for label in labels:
        result = all_results[label]
        
        psnr_vals = [result.get(k, {}).get("psnr", 0) for k in ["0", "1", "2"]]
        ssim_vals = [result.get(k, {}).get("ssim", 0) for k in ["0", "1", "2"]]

        plot_data["psnr"]["0"].append(psnr_vals[0])
        plot_data["psnr"]["1"].append(psnr_vals[1])
        plot_data["psnr"]["2"].append(psnr_vals[2])
        plot_data["psnr"]["all"].append(np.mean([v for v in psnr_vals if v > 0])) # 计算非零平均值

        plot_data["ssim"]["0"].append(ssim_vals[0])
        plot_data["ssim"]["1"].append(ssim_vals[1])
        plot_data["ssim"]["2"].append(ssim_vals[2])
        plot_data["ssim"]["all"].append(np.mean([v for v in ssim_vals if v > 0])) # 计算非零平均值

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制 PSNR 折线图
    #for key, values in plot_data["psnr"].items():
    ax1.plot(labels, plot_data["psnr"]['all'], marker='o', linestyle='-', label=f"Category {key}")
    ax1.set_title('PSNR vs. Iterations')
    ax1.set_xlabel('Iteration (Directory Name)')
    ax1.set_ylabel('Average PSNR')
    ax1.legend()
    ax1.grid(True)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # 绘制 SSIM 折线图
    for key, values in plot_data["ssim"].items():
        ax2.plot(labels, values, marker='o', linestyle='-', label=f"Category {key}")
    #ax2.plot(labels, plot_data["ssim"]['all'], marker='s', linestyle='--', label=f"Category {key}")
    ax2.set_title('SSIM vs. Iterations')
    ax2.set_xlabel('Iteration (Directory Name)')
    ax2.set_ylabel('Average SSIM')
    ax2.legend()
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()
    plt.savefig('val_result.png')

# --- 使用说明 ---
if __name__ == "__main__":
    # 请将这里的 'your_main_data_directory' 替换为您的实际数据根目录路径
    # 例如在 Windows 上: "D:\\data\\results"
    # 或在 Linux/macOS 上: "/home/user/data/results"

    process_and_plot_data("/home/linzhuohang/gso_outputs_v13")

