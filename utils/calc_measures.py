import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_image_metrics(img_path, src_path):
    """
    计算两张图片的PSNR和SSIM。

    Args:
        img_path (str): 生成图片的路径。
        src_path (str): 源图片的路径。

    Returns:
        dict: 包含 'psnr' 和 'ssim' 值的字典，如果出错则返回 None。
    """
    try:
        # 读取图片并转换为numpy数组
        img = np.array(Image.open(img_path))
        src_img = np.array(Image.open(src_path))

        # 确保图片尺寸一致
        if img.shape != src_img.shape:
            img.resize(256,256)
            src_img.resize(256,256)
        
        # 计算指标
        # data_range 是图像中像素值的可能范围
        psnr_val = psnr(src_img, img, data_range=255)
        
        # 对于多通道（彩色）图像，需要设置 channel_axis=-1
        channel_axis = -1 if img.ndim == 3 else None
        ssim_val = ssim(src_img, img, data_range=255, channel_axis=channel_axis)
        
        return {"psnr": psnr_val, "ssim": ssim_val}

    except FileNotFoundError:
        # 如果其中一张图片不存在，则跳过
        return None
    except Exception as e:
        print(f"计算指标时发生错误 '{img_path}', '{src_path}': {e}")
        return None

def process_images_and_plot(root_dir):
    """
    遍历目录，直接从图片计算指标，汇总数据，并绘制折线图。

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
        print(f"--- 正在处理文件夹: {dir_name} ---")
        
        # 用于存储当前大文件夹下所有小文件夹数据的总和与计数
        aggregated_data = {
            #"0": {"psnr": 0, "ssim": 0, "count": 0},
            "1": {"psnr": 0, "ssim": 0, "count": 0},
            "2": {"psnr": 0, "ssim": 0, "count": 0},
        }

        # 遍历小文件夹 (0 to 19)
        for i in range(11):
            sub_folder_path = os.path.join(main_folder_path, str(i))
            if not os.path.isdir(sub_folder_path):
                continue

            # 遍历每个类别 ("0", "1", "2")
            for key in aggregated_data.keys():
                img_path = os.path.join(sub_folder_path, f'{key}.jpg')
                src_path = os.path.join(sub_folder_path, f'src_{key}.jpg')

                metrics = calculate_image_metrics(img_path, src_path)
                metrics['psnr'] = metrics['psnr'] if metrics['psnr']!= np.inf else 100
                if metrics:
                    aggregated_data[key]["psnr"] += metrics["psnr"]
                    aggregated_data[key]["ssim"] += metrics["ssim"]
                    aggregated_data[key]["count"] += 1

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
                avg_results[key] = {"psnr": 0, "ssim": 0}
        
        all_results[dir_name] = avg_results
        
        print(all_results[dir_name])

    # --- 数据准备与绘图 (与之前脚本相同) ---
    if not all_results:
        print("没有处理任何数据，无法绘图。")
        return

    labels = sorted(all_results.keys(), key=int)
    
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
        plot_data["psnr"]["all"].append(np.mean([v for v in psnr_vals if v > 0]) if any(v > 0 for v in psnr_vals) else 0)

        plot_data["ssim"]["0"].append(ssim_vals[0])
        plot_data["ssim"]["1"].append(ssim_vals[1])
        plot_data["ssim"]["2"].append(ssim_vals[2])
        plot_data["ssim"]["all"].append(np.mean([v for v in ssim_vals if v > 0]) if any(v > 0 for v in ssim_vals) else 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    #for key, values in plot_data["psnr"].items():
    ax1.plot(labels, plot_data["psnr"]["all"], marker='o', linestyle='-', label=f"Category all")
    ax1.set_title('PSNR vs. Iterations')
    ax1.set_xlabel('Iteration (Directory Name)')
    ax1.set_ylabel('Average PSNR')
    ax1.legend()
    ax1.grid(True)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2.plot(labels, plot_data["ssim"]["all"], marker='o', linestyle='-', label=f"Category all")
    ax2.set_title('SSIM vs. Iterations')
    ax2.set_xlabel('Iteration (Directory Name)')
    ax2.set_ylabel('Average SSIM')
    ax2.legend()
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()
    plt.savefig('train_result.png')

# --- 使用说明 ---
if __name__ == "__main__":
    # 为了演示，我们创建一个假的目录结构和数据

    process_images_and_plot('/home/linzhuohang/train_outputs_v4')

