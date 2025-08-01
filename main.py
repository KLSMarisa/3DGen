#!/usr/bin/env python
# coding=utf-8
import os
import logging
from omegaconf import OmegaConf
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
# from trainer.trainer import *
from trainer.trainer_tri import *
from trainer.trainer_flux import Flux_Trainer
from data import create_dataloader, create_val_dataloader
# import pyvista as pv

os.environ['WORLD_SIZE']='1'
os.environ['LOCAL_RANK']='0'

# pv.start_xvfb()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='main', help='experiment identifier')
    parser.add_argument('--savedir', type=str, default='/mnt/nfs/caixiao/deeplearning/ckpt/3DGEN', help='path to save checkpoints and logs')
    parser.add_argument('--exp', type=str, default='diffusion', choices=['diffusion', 'renderer','dit','flux'], help='experiment type to run')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='experiment mode to run')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    """ Args about Data """
    parser.add_argument('--dataset', type=str, default='text2obj') # webvid
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--center_crop', default=False, action="store_true", help=('Whether to center crop the input images to the resolution.'))
    parser.add_argument('--random_flip', default=True, action='store_true', help='whether to randomly flip images horizontally')

    """ Args about Model """
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3], help='training stage for model')

    """ Args about Training """
    parser.add_argument('--nodes', type=int, default=1, help='nodes')
    parser.add_argument('--devices', type=int, default=1, help='e.g., gpu number')

    return parser.parse_args()




args = parse_args()
pl.seed_everything(args.seed, workers=True)
config = OmegaConf.load(args.config)
config.name = args.name
config.savedir = args.savedir
config.mode = args.mode
config.datasets = args.dataset
config.batch_size = args.batch_size
config.center_crop = args.center_crop
config.random_flip = args.random_flip
config.ddconfig.stage = args.stage



# os.environ['WORLD_SIZE'] = '8'

# print('MASTER_ADDR', os.environ['MASTER_ADDR'])
# print('MASTER_PORT', os.environ['MASTER_PORT'])
# print('LOCAL_RANK', os.environ['LOCAL_RANK'])
# print('WORLD_SIZE', os.environ['WORLD_SIZE'])
# exit()

lr_monitor = LearningRateMonitor(logging_interval='step')
### Define datasets
dataloader = create_dataloader(config)
# val_dataloader = create_val_dataloader(config)
### Define model
if args.exp == 'diffusion':
    trainer_model = StableDiffusionTrainer(config.ddconfig)
elif args.exp == 'renderer':
    trainer_model = RendererTrainer(config.renderconfig)
elif args.exp == 'dit':
    trainer_model = STDITTrainer(config.ddconfig)
elif args.exp == 'flux':
    trainer_model = Flux_Trainer()


### Define trainer
checkpoint_callback = ModelCheckpoint(
    dirpath                   =     os.path.join(config.savedir, 'checkpoints'),
    filename                  =     config.name + '-{step}', # -{epoch:02d}
    monitor                   =     'step',
    save_last                 =     False,
    save_top_k                =     -1,
    verbose                   =     True,
    every_n_train_steps       =     1000,
    save_on_train_epoch_end   =     True,
)

strategy = DeepSpeedStrategy(
    stage                     =     2, 
    offload_optimizer         =     True, 
    overlap_comm=False,
    logging_level=logging.DEBUG
    # offload_parameters        =     True,
    # offload_params_device     =     'cpu',
    # cpu_checkpointing         =     True,
)

trainer = pl.Trainer(
    default_root_dir          =     config.savedir,
    callbacks                 =     [checkpoint_callback, lr_monitor, ModelSummary(2)], # ModelSummary(2)
    accelerator               =     'gpu',
    accumulate_grad_batches   =     config.gradient_accumulation_steps,
    # accumulate_grad_batches   =     5,
    benchmark                 =     True,
    num_nodes                 =     args.nodes,
    devices                   =     args.devices,
    gradient_clip_val         =     config.max_grad_norm,
    log_every_n_steps         =     1,
    precision                 =     'bf16-mixed', #"bf16",
    max_epochs                =     config.num_train_epochs,
    strategy                  =     strategy,
    sync_batchnorm            =     True,
    # val_check_interval        =     100,
    check_val_every_n_epoch=None,
    
)

if args.mode == 'train':
    ### training
    trainer.fit(
        model                     =     trainer_model,
        train_dataloaders         =     dataloader,
        # val_dataloaders           =     val_dataloader,
        ckpt_path                 =     None if not os.path.exists(args.resume) else args.resume,
    )
elif args.mode == 'test':
    assert os.path.exists(args.resume), "resume path does not exist"
    trainer.test(
        model                     =     trainer_model,
        test_dataloaders          =     dataloader,
        ckpt_path                 =     args.resume,
    )

