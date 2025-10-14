# server_broadcast.py
import argparse
import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from starlette.responses import StreamingResponse
from PIL import Image
import io
import uvicorn
import asyncio
import os
import multiprocessing
import uuid
from typing import Dict, List

# --- 1. 全局配置 ---
MAX_GPUS_TO_USE = 6
MODEL_PATH = "/mnt/hdd3/linzhuohang/3DGen/hf/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/af58063aa431f4d2bbc11ae46f57451d4416a170"
LORA_PATH = '/mnt/hdd3/linzhuohang/3DGen/hf/hub/models--JD3GEN--JD3_Nudify_Kontext_LoRa/snapshots/c4206e2598d821a790081479a27b254af64e5c86'

# --- 2. FastAPI应用和全局状态 ---
app = FastAPI()
# 用于在主进程中等待特定请求结果的事件通知
request_events: Dict[str, asyncio.Event] = {}
# 存储从结果队列收到的最终数据 (现在每个ID对应一个结果列表)
request_results: Dict[str, List[bytes]] = {}


# --- 3. 消费者：模型工作进程 (代码本身无需修改) ---
def model_worker(gpu_id, task_q, result_q, model_path, lora_path):
    device = f"cuda:{gpu_id}"
    print(f"[Worker-{gpu_id}] 进程 {os.getpid()} 启动，将在 {device} 上工作。")

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        #pipe.load_lora_weights(lora_path)
        pipe = pipe.to(device)
        print(f"[Worker-{gpu_id}] 模型加载成功。")
    except Exception as e:
        print(f"!!! [Worker-{gpu_id}] 模型加载失败: {e}")
        return

    while True:
        try:
            # 每个worker从自己的专属队列中获取任务
            request_id, image_bytes, prompt = task_q.get()
            
            if request_id is None:
                print(f"[Worker-{gpu_id}] 收到终止信号，即将退出。")
                break

            print(f"[Worker-{gpu_id}] 开始处理请求 {request_id}")
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            generated_image = pipe(
                image=input_image, prompt=prompt, guidance_scale=2.5,
            ).images[0]
            img_byte_arr = io.BytesIO()
            generated_image.save(img_byte_arr, format='PNG')
            # 将结果(带上自己的gpu_id)放入共享的结果队列
            result_q.put((request_id, gpu_id, img_byte_arr.getvalue()))
            print(f"[Worker-{gpu_id}] 完成处理请求 {request_id}")

        except Exception as e:
            print(f"!!! [Worker-{gpu_id}] 处理任务时发生错误: {e}")
            result_q.put((request_id, gpu_id, f"Error: {e}"))
        finally:
            torch.cuda.empty_cache()

    del pipe
    print(f"[Worker-{gpu_id}] 进程退出。")


# --- 4. 主进程中的异步结果收集器 (逻辑修改) ---
async def result_collector(result_q, num_workers):
    """
    现在需要收集齐 num_workers 个结果后才触发事件。
    """
    print("[Main] 结果收集器已启动。")
    while True:
        if not result_q.empty():
            request_id, gpu_id, result_data = result_q.get()
            
            if request_id in request_events:
                # 初始化该请求的结果列表 (如果尚未存在)
                if request_id not in request_results:
                    request_results[request_id] = []
                
                # 添加结果
                request_results[request_id].append(result_data)
                print(f"[Main] 收到请求 {request_id} 的 {len(request_results[request_id])}/{num_workers} 个结果。")

                # 如果收集齐了所有worker的结果，则触发事件
                if len(request_results[request_id]) == num_workers:
                    print(f"[Main] 请求 {request_id} 的所有结果已收集完毕。")
                    request_events[request_id].set()
        else:
            await asyncio.sleep(0.01)


# --- 5. FastAPI 应用生命周期事件 (逻辑修改) ---
@app.on_event("startup")
async def startup_event():
    available_gpus = torch.cuda.device_count()
    num_gpus = min(MAX_GPUS_TO_USE, available_gpus)
    app.state.num_workers = num_gpus

    if num_gpus == 0:
        print("!!! 错误：未检测到CUDA设备，服务无法启动工作进程。")
        return

    print(f"--- 正在为 {num_gpus} 个GPU启动工作进程... ---")
    
    # 为每个worker创建一个专属的任务队列
    app.state.task_queues = [app.state.manager.Queue() for _ in range(num_gpus)]
    app.state.worker_processes = []
    
    result_queue = app.state.result_queue
    
    for i in range(num_gpus):
        # 将专属任务队列传递给对应的worker
        task_queue = app.state.task_queues[i]
        p = multiprocessing.Process(
            target=model_worker,
            args=(i, task_queue, result_queue, MODEL_PATH, LORA_PATH),
            daemon=True
        )
        app.state.worker_processes.append(p)
        p.start()
    
    # 启动结果收集器时告知总worker数量
    asyncio.create_task(result_collector(result_queue, num_gpus))
    print("--- 所有工作进程和结果收集器已启动。服务准备就绪。 ---")

@app.on_event("shutdown")
def shutdown_event():
    print("--- 服务正在关闭，开始终止工作进程... ---")
    task_queues = app.state.task_queues
    worker_processes = app.state.worker_processes
    
    # 向每个专属任务队列发送终止信号
    for q in task_queues:
        q.put((None, None, None))
    
    for p in worker_processes:
        p.join(timeout=5)
        if p.is_alive(): p.terminate()

    print("--- 所有工作进程已终止。 ---")


# --- 6. API 端点 (逻辑修改) ---
@app.post("/generate_single")
async def generate_single(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    if not request.app.state.worker_processes:
        raise HTTPException(status_code=503, detail="没有可用的工作进程。")

    try:
        request_id = str(uuid.uuid4())
        image_bytes = await image.read()
        event = asyncio.Event()
        request_events[request_id] = event

        # **关键修改**: 广播任务到所有worker的专属队列
        print(f"[Main] 广播请求 {request_id} 到 {request.app.state.num_workers} 个工作进程。")
        task_queues = request.app.state.task_queues
        for q in task_queues:
            q.put((request_id, image_bytes, prompt))

        await asyncio.wait_for(event.wait(), timeout=300)

        results = request_results[request_id]
        
        # 处理结果，检查是否有错误
        images_bytes_list = []
        for res in results:
            if isinstance(res, str) and res.startswith("Error:"):
                raise HTTPException(status_code=500, detail=f"某个工作进程出错: {res}")
            images_bytes_list.append(res)
        
        if not images_bytes_list:
            raise HTTPException(status_code=500, detail="所有工作进程都未能成功生成图像。")

        # **新增**: 将所有图片拼接成一张
        pil_images = [Image.open(io.BytesIO(b)) for b in images_bytes_list]
        
        widths, heights = zip(*(i.size for i in pil_images))
        total_width = sum(widths)
        max_height = max(heights)

        # 创建一个新的空白图片用于拼接
        stitched_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in pil_images:
            stitched_image.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        # 将拼接后的大图保存到内存
        final_image_bytes = io.BytesIO()
        stitched_image.save(final_image_bytes, format='PNG')
        final_image_bytes.seek(0)
        
        return StreamingResponse(final_image_bytes, media_type="image/png")

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="请求处理超时，未能收集到所有GPU的结果。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生未知错误: {e}")
    finally:
        if request_id in request_events: del request_events[request_id]
        if request_id in request_results: del request_results[request_id]


# --- 7. 服务启动入口 (逻辑修改) ---
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # 创建Manager并附加到app.state，以便在整个应用中共享
    manager = multiprocessing.Manager()
    app.state.manager = manager
    # 现在只创建共享的结果队列
    app.state.result_queue = manager.Queue()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=4)
    args = parser.parse_args()
    MAX_GPUS_TO_USE = args.gpus
    
    uvicorn.run(app, host="0.0.0.0", port=6666)