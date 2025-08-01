import flux_modules.env_set
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from diffusers.pipelines.pipeline_utils import PipelineQuantizationConfig
import accelerate
print("--- 导入成功 ---")


def run_pipe():
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
    )
    torch.cuda.empty_cache()
    pipe.enable_attention_slicing()
    #pipe.enable_sequential_cpu_offload()
    pipe = pipe.to("cuda")
    print("--- 模型加载成功 ---")
    image = load_image("sample.jpg").convert("RGB")
    prompt = "change the person to kiresame marisa in touhou project."
    print("--- 开始生成图像 ---")

    image = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=2.5,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
    print("--- 图像生成完成 ---")
    image.save("flux-kontext.png")

if __name__ == "__main__":
    run_pipe()
    print("--- 运行完成 ---")