import subprocess
import time
import os
import re
import logging
import signal
import sys
import argparse
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpu_monitor2.log"),
        logging.StreamHandler()
    ]
)

# 全局变量，用于优雅退出
exit_flag = False

def signal_handler(sig, frame):
    """处理退出信号"""
    global exit_flag
    logging.info("接收到终止信号，准备退出...")
    exit_flag = True

def get_gpu_memory_usage():
    """使用nvidia-smi获取GPU内存使用情况（MB）"""
    try:
        # 使用 check=True 确保在命令失败时抛出异常
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        gpu_memory = {}
        for line in result.stdout.strip().split('\n'):
            match = re.search(r'(\d+),\s*(\d+)', line)
            if match:
                gpu_id = int(match.group(1))
                memory_used = int(match.group(2))
                gpu_memory[gpu_id] = memory_used
        return gpu_memory
    except subprocess.CalledProcessError as e:
        logging.error(f"nvidia-smi执行失败: {e.stderr}")
    except Exception as e:
        logging.error(f"获取GPU内存使用失败: {str(e)}")
    return {}

def find_free_gpus(threshold=1000):
    """查找内存使用低于阈值（默认为 10000 MB）的GPU"""
    gpu_memory = get_gpu_memory_usage()
    free_gpus = []
    for gpu_id, memory_used in gpu_memory.items():
        if memory_used < threshold:
            free_gpus.append(gpu_id)
            logging.debug(f"GPU {gpu_id}: {memory_used} MB (空闲)")
        else:
            logging.debug(f"GPU {gpu_id}: {memory_used} MB (占用)")
    free_gpus.sort()
    return free_gpus

def launch_process(gpu_indices, script_path):
    """设置环境变量并启动脚本或 Python 文件，等待其完成"""
    try:
        device_str = ",".join(map(str, gpu_indices))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        logging.info(f"设置 CUDA_VISIBLE_DEVICES={device_str}")
        log_dir = "task_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"task_{timestamp}.log")
        logging.info(f"启动脚本: {script_path}, 日志文件: {log_file}")
        
        # 判断是否为 Python 文件，构造执行命令
        command = ["python", script_path] if script_path.endswith(".py") else [script_path]
        
        with open(log_file, "w") as log_f:
            log_f.write(f"任务启动时间: {timestamp}\n")
            log_f.write(f"使用的GPU: {device_str}\n")
            log_f.write("-" * 50 + "\n")
            process = subprocess.Popen(
                command,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=os.environ,
                text=True
            )
            return_code = process.wait()
            log_f.write("\n" + "-" * 50 + "\n")
            log_f.write(f"任务结束时间: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_f.write(f"退出代码: {return_code}\n")
            
            if return_code == 0:
                logging.info("脚本执行成功完成")
            else:
                logging.error(f"脚本执行失败，退出代码: {return_code}")
        return return_code
    except Exception as e:
        logging.error(f"启动任务失败: {str(e)}")
        return 1

def monitor_gpus(required_gpus, script_path, continue_on_failure):
    """监控GPU内存使用情况，并在找到足够资源时启动任务"""
    global exit_flag
    try:
        # 检查 nvidia-smi 可用性
        subprocess.run(['nvidia-smi', '-L'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info("nvidia-smi 可用，开始监控GPU...")
        
        # 获取 GPU 数量
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, text=True)
        gpu_count = len(result.stdout.strip().split('\n')) if result.stdout else 0
        if gpu_count == 0:
            logging.error("未检测到GPU设备，程序退出。")
            return
        else:
            logging.info(f"检测到 {gpu_count} 个GPU设备")

        if required_gpus > gpu_count:
            logging.error(f"所需 GPU 数量 ({required_gpus}) 大于系统 GPU 总数 ({gpu_count})，请检查参数。")
            sys.exit(1)

        while not exit_flag:
            free_gpus = find_free_gpus()
            
            if len(free_gpus) >= required_gpus:
                logging.info(f"找到 {len(free_gpus)} 张空闲GPU: {free_gpus}。准备启动任务...")
                
                # 启动任务
                return_code = launch_process(free_gpus[:required_gpus], script_path)

                if return_code == 0:
                    # 任务成功，退出程序
                    logging.info("任务执行成功，程序退出。")
                    sys.exit(0)
                else:
                    # 任务失败
                    logging.error(f"任务执行失败，退出代码: {return_code}")
                    
                    if continue_on_failure:
                        # 开启重试模式
                        logging.warning("检测到 --continue 标志，等待下一轮监控以重试任务...")
                        # 循环将继续，等待 30 秒后再次检查
                    else:
                        # 非重试模式，退出程序
                        logging.error("未设置 --continue 标志，任务失败后程序退出。")
                        sys.exit(1)
            
            else:
                logging.info(f"当前空闲GPU数量不足: {len(free_gpus)}/{required_gpus}，继续监控...")
                
            # 等待 30 秒或直到接收到退出信号
            for _ in range(30):
                if exit_flag:
                    break
                time.sleep(1)
                
    except FileNotFoundError:
        logging.error("nvidia-smi 命令未找到，请确保NVIDIA驱动已安装")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logging.error("无法访问GPU信息，请检查nvidia-smi权限")
        sys.exit(1)
    except Exception as e:
        logging.error(f"监控过程中出错: {str(e)}")
        sys.exit(1)
    finally:
        logging.info("GPU监控已停止")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU监控脚本，等待资源空闲后启动任务。")
    parser.add_argument("--gpus", type=int, default=4, help="需要使用的GPU数量 (默认: 4)")
    parser.add_argument("--script", type=str, required=True, help="要运行的脚本路径 (例如: /path/to/train.py)")
    parser.add_argument(
        "--continue",
        dest="continue_on_failure",
        action="store_true",
        default=False,
        help="如果任务失败，是否继续监控GPU资源并重试启动任务 (默认: False)"
    )
    
    args = parser.parse_args()
    
    # 信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("GPU空闲资源监控系统启动")
    
    # 确保脚本可执行
    if args.script.endswith('.sh'):
        try:
            os.system('chmod +x ' + args.script)
            logging.info(f"已为 {args.script} 添加执行权限。")
        except Exception as e:
            logging.warning(f"无法设置脚本执行权限: {e}")
            
    monitor_gpus(args.gpus, args.script, args.continue_on_failure)
    
    logging.info("程序退出")
