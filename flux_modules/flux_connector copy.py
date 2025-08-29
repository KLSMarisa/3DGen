import env_set
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import accelerate
print("--- 导入成功 ---")


def run_pipe():
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    torch.cuda.empty_cache()
    pipe.enable_attention_slicing()
    #pipe.enable_sequential_cpu_offload()
    pipe = pipe.to("cuda")
    print("--- 模型加载成功 ---")
    image = load_image("/home/linzhuohang/3DGen/flux_modules/sample.jpg").convert("RGB")
    prompt = "generate the side view of the object"
    print("--- 开始生成图像 ---")

    image = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=2.5,
    ).images[0]
    print("--- 图像生成完成 ---")
    image.save("flux-kontext_0.png")
    

if __name__ == "__main__":
    run_pipe()
    print("--- 运行完成 ---")