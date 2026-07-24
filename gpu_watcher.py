import subprocess
import time
import os
import re
import logging
import signal
import sys
import argparse
from datetime import datetime
THRESHOLD=1000
STALE_TIMEOUT = 600  # 10 minutes in seconds
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

def find_free_gpus(threshold=THRESHOLD):
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

def select_best_topology_gpus(free_gpus, count):
    """
    从空闲GPU中选择指定数量的GPU，优先选择处于同一组（(0,1), (2,3)...）的GPU。
    """
    if len(free_gpus) < count:
        return []
    
    # 将GPU按组分类，组ID = gpu_id // 2
    groups = {}
    for gpu in free_gpus:
        group_id = gpu // 2
        groups.setdefault(group_id, []).append(gpu)
        
    full_pairs = []
    singles = []
    
    # 区分完整对和单个GPU
    for gid in sorted(groups.keys()):
        g_gpus = sorted(groups[gid])
        if len(g_gpus) == 2:
            full_pairs.append(g_gpus)
        else:
            singles.extend(g_gpus)
            
    selected = []
    needed = count
    
    # 1. 优先获取完整对
    while needed >= 2 and full_pairs:
        pair = full_pairs.pop(0)
        selected.extend(pair)
        needed -= 2
        
    # 2. 如果还需要，从剩余池中获取（包括单个GPU和未被选中的完整对）
    remaining_pool = singles
    for pair in full_pairs:
        remaining_pool.extend(pair)
    remaining_pool.sort()
    
    if needed > 0:
        selected.extend(remaining_pool[:needed])
        
    return sorted(selected)

def launch_process(gpu_indices, script_path, monitor=False, stale_timeout=STALE_TIMEOUT, check_interval=30):
    """设置环境变量并启动脚本或 Python 文件，等待其完成。
    如果 monitor=True，则监视日志文件的修改时间；在超过 stale_timeout 秒没有更新时重启进程。
    返回最终进程的退出代码。
    """
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

        # Helper to start the subprocess and append to the same log file
        def _start_process():
            log_f = open(log_file, "a", buffering=1)
            log_f.write(f"\n任务启动时间: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_f.write(f"使用的GPU: {device_str}\n")
            log_f.write("-" * 50 + "\n")
            proc = subprocess.Popen(
                command,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=os.environ,
                text=True
            )
            return proc, log_f

        # Start the first process
        proc, log_f = _start_process()
        last_update = time.time()
        try:
            # Initialize last_update to the current mtime if file exists
            if os.path.exists(log_file):
                last_update = os.path.getmtime(log_file)
                # convert to epoch
                if isinstance(last_update, float):
                    last_update = last_update
                last_update = time.time()

            while True:
                # Poll process
                ret = proc.poll()

                # Check log modification time
                try:
                    mtime = os.path.getmtime(log_file)
                    # If file mtime changed recently, update last_update timestamp
                    # We compare against previous recorded modification via time() to track activity.
                    if time.time() - mtime < check_interval + 1:
                        last_update = time.time()
                except Exception:
                    # If we can't stat the file, ignore and continue
                    pass

                # If monitoring is enabled and log is stale, restart process
                if monitor and (time.time() - last_update) > stale_timeout:
                    logging.warning(f"日志在 {stale_timeout} 秒内无更新，重启任务...")
                    # Try graceful terminate
                    try:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    except Exception as e:
                        logging.warning(f"终止进程时出错: {e}")

                    # Close old log handle and write restart marker
                    try:
                        log_f.write("\n" + "-" * 50 + "\n")
                        log_f.write(f"日志无更新，重启时间: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                        log_f.flush()
                        log_f.close()
                    except Exception:
                        pass

                    # Restart process
                    proc, log_f = _start_process()
                    last_update = time.time()

                # If process exited, finalize and return code
                if ret is not None:
                    try:
                        log_f.write("\n" + "-" * 50 + "\n")
                        log_f.write(f"任务结束时间: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                        log_f.write(f"退出代码: {ret}\n")
                        log_f.close()
                    except Exception:
                        pass

                    if ret == 0:
                        logging.info("脚本执行成功完成")
                    else:
                        logging.error(f"脚本执行失败，退出代码: {ret}")
                    return ret

                # Sleep a short while before next check
                for _ in range(check_interval):
                    if exit_flag:
                        break
                    time.sleep(1)

                if exit_flag:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        log_f.write("\n" + "-" * 50 + "\n")
                        log_f.write(f"因接收到退出信号，任务被终止: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                        log_f.close()
                    except Exception:
                        pass
                    return 1

        finally:
            # Ensure file handle closed
            try:
                if not log_f.closed:
                    log_f.close()
            except Exception:
                pass

    except Exception as e:
        logging.error(f"启动任务失败: {str(e)}")
        return 1

def monitor_gpus(required_gpus, script_path, continue_on_failure, monitor_flag=False):
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
                # 使用优先分组策略选择 GPU
                best_gpus = select_best_topology_gpus(free_gpus, required_gpus)
                # monitor only takes effect when used together with continue_on_failure
                effective_monitor = monitor_flag and continue_on_failure
                if monitor_flag and not continue_on_failure:
                    logging.warning("--monitor 已启用但未设置 --continue；忽略监控重启行为，继续正常启动任务。")
                return_code = launch_process(best_gpus, script_path, monitor=effective_monitor)

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
    parser.add_argument(
        "--monitor",
        dest="monitor",
        action="store_true",
        default=False,
        help="配合 --continue 使用：持续监控任务日志，超过10分钟无更新则重启任务"
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
            
    monitor_gpus(args.gpus, args.script, args.continue_on_failure, args.monitor)
    
    logging.info("程序退出")
