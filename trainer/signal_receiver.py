import signal
import os
from pytorch_lightning import LightningModule, Trainer, Callback

# 假设你已经定义好了你的模型和数据模块
# from your_model_file import MyLitModel
# from your_data_file import MyDataModule

class CheckpointOnInterrupt(Callback):
    """
    在接收到Ctrl+C (KeyboardInterrupt) 时保存检查点。
    """
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.interrupted = False
        self.original_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self.interrupt_handler)

    def interrupt_handler(self, signum, frame):
        """
        自定义信号处理函数。
        """
        print("\n捕获到 Ctrl+C 信号！正在准备保存检查点并退出...")
        self.interrupted = True
        # 恢复原始的信号处理器，以允许程序在保存后能正常退出
        signal.signal(signal.SIGUSR1, self.original_handler)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        在每个训练批次结束后检查中断标志。
        """
        if self.interrupted:
            # 标记为True后，我们在此处保存模型
            checkpoint_path = self.save_path
            trainer.model.transformer.save_pretrained(checkpoint_path,max_shard_size="3GB",safe_serialization=True)
            print(f"检查点已保存至: {checkpoint_path}")
            # 重新抛出KeyboardInterrupt以正常终止训练
            raise KeyboardInterrupt

    def on_train_end(self, trainer, pl_module):
        """
        训练正常结束时，恢复原始信号处理器。
        """
        signal.signal(signal.SIGUSR1, self.original_handler)