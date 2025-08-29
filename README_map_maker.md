# Map Maker 脚本使用说明

## 功能特性

- ✅ **实时日志记录**: 程序运行时既在终端显示又保存到日志文件
- ✅ **后台运行模式**: 类似screen功能，程序可在后台运行
- ✅ **自动日志命名**: 默认使用时间戳和程序名生成日志文件
- ✅ **自定义日志名**: 用户可指定日志文件名
- ✅ **PID管理**: 后台运行时自动管理进程ID
- ✅ **GPU状态监控**: 自动检查和记录GPU使用情况
- ✅ **完整参数验证**: 检查所有必需参数和路径

## 使用方法

### 1. 基本前台运行
```bash
./map_maker.sh
```
使用默认参数运行，实时显示输出并保存日志。

### 2. 指定自定义日志名
```bash
./map_maker.sh --log_name "llava_experiment_v1"
```
日志将保存为 `./logs/llava_experiment_v1.log`

### 3. 后台运行模式
```bash
./map_maker.sh --background
```
程序将在后台运行，不会占用终端。

### 4. 后台运行 + 自定义日志
```bash
./map_maker.sh --background --log_name "background_test"
```

### 5. 完整参数示例
```bash
./map_maker.sh \
    --vanilla_dir /path/to/your/model \
    --model_id llava-hf/llava-1.5-7b-hf \
    --dataset_name MMMU_Pro \
    --batch_size 2 \
    --log_name "low_batch_experiment" \
    --background
```

## 日志功能

### 日志文件位置
- 默认目录: `./logs/`
- 可通过 `--log_dir` 参数自定义

### 默认日志命名规则
如果不指定 `--log_name`，日志文件名格式为:
```
map_maker_YYYYMMDD_HHMMSS.log
```
例如: `map_maker_20241201_143022.log`

### 日志内容包含
- 带时间戳的所有程序输出
- GPU状态信息
- 运行参数记录
- 开始和结束时间
- 任务执行状态

## 后台运行管理

### 方法一：使用内置管理命令 (推荐)

#### 启动后台任务
```bash
./map_maker.sh --background --log_name "my_task"
```

#### 查看任务状态
```bash
./map_maker.sh --status
```

#### 实时查看日志
```bash
./map_maker.sh --logs
```

#### 停止后台任务
```bash
./map_maker.sh --stop
```

#### 列出所有日志文件
```bash
./map_maker.sh --list
```

### 方法二：使用交互式管理器
```bash
./map_manager.sh
```

这将启动一个友好的交互式菜单，提供以下功能：
- 查看后台任务状态
- 实时查看日志
- 停止后台任务  
- 列出所有日志文件
- 启动新的后台任务
- 清理旧日志文件

### 方法三：手动管理
```bash
# 查看PID文件
cat ./logs/map_maker.pid

# 检查进程是否存在
ps aux | grep map_maker

# 手动停止
kill $(cat ./logs/map_maker.pid)
```

## 所有可用参数

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--vanilla_dir` | 预训练模型目录路径 | 已预设 | 是 |
| `--model_id` | Huggingface模型ID | `llava-hf/llava-1.5-7b-hf` | 是 |
| `--dataset_name` | 数据集名称 | `MMMU_Pro` | 否 |
| `--batch_size` | 批次大小 | `4` | 否 |
| `--max_length` | 最大序列长度 | `4000` | 否 |
| `--select_percent` | 神经元选择百分比 | `2` | 否 |
| `--model_type` | 模型类型 | `Llava` | 否 |
| `--save_path` | 结果保存路径 | `./global_map` | 否 |
| `--gpu_id` | 指定GPU设备ID | 自动选择 | 否 |
| `--log_dir` | 日志保存目录 | `./logs` | 否 |
| `--log_name` | 日志文件名 | 自动生成 | 否 |
| `--background` | 后台运行模式 | `false` | 否 |
| `--status` | 查看后台任务状态 | - | 否 |
| `--logs` | 实时查看日志 | - | 否 |
| `--stop` | 停止后台任务 | - | 否 |
| `--list` | 列出所有日志文件 | - | 否 |

## 故障排除

### 1. 权限问题
```bash
chmod +x map_maker.sh
```

### 2. 日志目录不存在
脚本会自动创建 `./logs` 目录，无需手动创建。

### 3. 后台进程无响应
```bash
# 强制终止所有相关进程
pkill -9 -f "map_maker"
```

### 4. 查看详细错误信息
```bash
# 查看日志文件末尾
tail -20 ./logs/your_log_file.log
```

## 快速开始指南

### 1. 启动后台任务
```bash
# 使用默认设置后台运行
./map_maker.sh --background

# 或指定日志名称
./map_maker.sh --background --log_name "experiment_1"
```

### 2. 监控任务
```bash
# 查看任务状态
./map_maker.sh --status

# 实时查看日志 (按Ctrl+C退出)
./map_maker.sh --logs
```

### 3. 管理任务
```bash
# 停止任务
./map_maker.sh --stop

# 查看所有日志文件
./map_maker.sh --list
```

### 4. 使用交互式管理器
```bash
./map_manager.sh
```

## 示例输出

### 查看状态时的输出:
```
=== Map Maker 后台任务状态 ===
[INFO] 后台任务正在运行中
[INFO] 进程ID: 12345
[INFO] PID文件: ./logs/map_maker.pid
[INFO] 最新日志文件: ./logs/map_maker_20241201_143022.log
[INFO] 日志文件大小: 2.3M

最近的日志内容 (最后10行):
----------------------------------------
[2024-12-01 14:45:22] Processing batch 45/120...
[2024-12-01 14:45:25] GPU Memory: 8.2GB/24GB
[2024-12-01 14:45:28] Collecting activations...
----------------------------------------
```

### 前台运行时的终端输出:
```
[INFO] [2024-12-01 14:30:22] 日志文件: ./logs/map_maker_20241201_143022.log
[INFO] [2024-12-01 14:30:22] 检查GPU状态...
[INFO] [2024-12-01 14:30:23] 运行参数:
[2024-12-01 14:30:23]   模型目录: /path/to/model
[2024-12-01 14:30:23]   模型ID: llava-hf/llava-1.5-7b-hf
...
[INFO] [2024-12-01 14:30:25] 开始生成神经元映射...
```

### 后台运行时的输出:
```
[INFO] [2024-12-01 14:30:22] 后台模式启动，PID文件: ./logs/map_maker.pid
[INFO] [2024-12-01 14:30:22] 可以使用以下命令查看日志:
[2024-12-01 14:30:22]   tail -f ./logs/map_maker_20241201_143022.log
[INFO] [2024-12-01 14:30:23] 后台进程已启动，PID: 12345
[INFO] [2024-12-01 14:30:23] 进程正在后台运行中...
[INFO] [2024-12-01 14:30:23] 实时查看日志: tail -f ./logs/map_maker_20241201_143022.log
```
