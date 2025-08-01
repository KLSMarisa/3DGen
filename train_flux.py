# train.py
import torch
from datasets import load_dataset
from flux_modules import OAFluxKontextPipeline2 as OAFluxKontextPipeline
from data import text2obj_dataset
import os
from trainer.trainer_flux import Flux_Trainer
os.environ['WORLD_SIZE']='1'
os.environ['LOCAL_RANK']='0'
# 1. 加载模型和分词器
model_name = "oa_flux_transformer"
# 必须要有 pad_token
train_dataset = text2obj_dataset.load_text2obj_dataset(data_dir='configs/select_data1.json')
val_dataset = text2obj_dataset.load_text2obj_dataset(data_dir='configs/select_data2.json')

trainer = Flux_Trainer()


# 2. 加载和预处理数据集