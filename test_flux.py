
import argparse
import flux_modules.OAFluxKontextPipeline2 as FluxKontextPipeline
import PIL.Image as Image
import torch
def generate_sample(step):
    try:
        image = Image.open('/home/linzhuohang/3DGen/data/000.png').convert('RGB')
        pipeline = FluxKontextPipeline.get_pipeline(Train = False,ckpt_path=None)
        for i,block in enumerate(pipeline.transformer.transformer_blocks):
            block.enable_oa = True
        prompt = 'A bird open its wings'
        result =  pipeline(image,prompt=prompt,prompt_2 = prompt,height=256,width=256)
        for i in range(3):
            result[i].save(f'/home/linzhuohang/3DGen/train_outputs/sample_{i}.jpg')
        print('completed')
    except Exception as e:
        raise e
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=6000, help='step number for the model')
    args = parser.parse_args()
    generate_sample(args.step)
    print('Sample generation completed.')