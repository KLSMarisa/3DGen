from pytorch_lightning.callbacks import Callback
import torch
class ImageLogger(Callback):
    def __init__(self, image_interval):
        super().__init__()
        self.image_interval = image_interval

    # 在每个训练批次结束后调用
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # trainer.global_step 是从 0 开始的
        # local_rank==0 确保只有主进程保存图片
        if (trainer.global_step) % self.image_interval == 0 and trainer.local_rank == 0:
            print(f"global_step: {trainer.global_step}, generating sample")
            
            # 使用 pl_module 引用你的训练器实例
            # 将模型切换到评估模式，并使用 no_grad
            pl_module.eval()
            with torch.no_grad():
                result = pl_module.inference(batch['img'][0].unsqueeze(0), batch['caption'][0])
            pl_module.train() # 切换回训练模式

            for i in range(len(result)):
                result[i].save(f'/home/linzhuohang/train_outputs/{trainer.global_step}_{i}.jpg')
