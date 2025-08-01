import torch
import deepspeed
from PIL import Image
from flux_modules.OAFluxKontextPipeline2 import get_pipeline
import flux_modules.env_set

def main():
    # 初始化DeepSpeed
    with torch.no_grad():
        deepspeed.init_distributed()
        
        # 获取pipeline
        pipe = get_pipeline()
        pipe = pipe.to("cuda")
        # 加载和准备图像
        init_image = Image.open("/home/linzhuohang/3DGen/flux_modules/sample.jpg").convert("RGB")
        prompt = "kirisame marisa in touhou project, highly detailed, vibrant colors"
        
        # 运行推理
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images = pipe(
                prompt=prompt,
                prompt_2= prompt,
                image=init_image,
                num_inference_steps=50,
                height=512,
                width=512,
            ).images
        
        # 保存结果
        for i, img in enumerate(images):
            img.save(f"deepspeed_output_{i}.png")

if __name__ == "__main__":
    main()
