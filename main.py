#!/usr/bin/env python
# coding=utf-8
import datetime
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
#from diffusers import FluxKontextPipeline 
import accelerate 
import deepspeed
#from flux_modules import OAFluxKontextPipeline2 as OAFluxKontextPipeline
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.elastic.multiprocessing.errors import record
# from trainer.trainer import *
#from trainer.trainer_tri import *
#from trainer.trainer_flux import Flux_Trainer
#from trainer.trainer_pose2 import Pose_Trainer

from data import create_dataloader, create_val_dataloader
import time
# import pyvista as pv
from trainer.image_logger import ImageLogger
from trainer.signal_receiver import CheckpointOnInterrupt
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.io import AsyncCheckpointIO
import re
#os.environ['WORLD_SIZE'] = '1'
#os.environ['LOCAL_RANK'] = '0'
class ResetOptimizerCallback(Callback):
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # 确保在恢复训练时才执行
        print("Resetting optimizer state from checkpoint.")
        # 从加载的检查点字典中移除优化器和调度器状态
        checkpoint.pop('optimizer_states', None)
        checkpoint.pop('lr_schedulers', None)

# 使用这个回调

# pv.start_xvfb()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='main', help='experiment identifier')
    parser.add_argument('--savedir', type=str, default='/mnt/nfs/caixiao/deeplearning/ckpt/3DGen', help='path to save checkpoints and logs')
    parser.add_argument('--exp', type=str, default='diffusion', help='experiment type to run')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test','inference','validate'], help='experiment mode to run')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    """ Args about Data """
    parser.add_argument('--dataset', type=str, default='text2obj') # webvid
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--center_crop', default=False, action="store_true", help=('Whether to center crop the input images to the resolution.'))
    parser.add_argument('--random_flip', default=True, action='store_true', help='whether to randomly flip images horizontally')

    """ Args about Model """
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--init_step', type=int, default='0')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3], help='training stage for model')

    """ Args about Training """
    parser.add_argument('--nodes', type=int, default=1, help='nodes')
    parser.add_argument('--devices', type=int, default=1, help='e.g., gpu number')
    parser.add_argument('--version', type=int, default=1, help='version of trainning')
    parser.add_argument('--every_n_train_steps', type=int, default=500, help='every n train steps to save checkpoint')
    parser.add_argument('--grad_acc', type=int, default=1, help='accumulate grad batches')
    
    #parser.add_argument('--local-rank',default=0)
    return parser.parse_args()

@record
def main_func():

    args = parse_args()
    print(args)
    if(args.seed==-1): args.seed=time.time()
    
    config = OmegaConf.load(args.config)
    config.name = args.name
    config.version = args.version
    config.savedir = os.path.join(config.data_dir , f'ckptv{config.version}')
    config.mode = args.mode
    config.datasets = args.dataset
    config.batch_size = args.batch_size
    config.center_crop = args.center_crop
    config.random_flip = args.random_flip
    config.ddconfig.stage = args.stage
    if args.resume=='latest':
        print('searching for latest checkpoint...')
        ckpt_dir = f'{config.savedir}/checkpoints'
        print('searching in ',ckpt_dir)
        latest_step = 0
        if os.path.exists(ckpt_dir):
            pattern = re.compile(f'{args.name}-step=(\d+)\.ckpt')
            steps = []
            for fname in os.listdir(ckpt_dir):
                m = pattern.match(fname)
                if m:
                    steps.append(int(m.group(1)))
                if steps:
                    latest_step = max(steps)
        args.resume=f'{config.savedir}/checkpoints/{args.name}-step={latest_step}.ckpt'
        print(f'found checkpoint for step {latest_step}, resuming from {args.resume}')
    pl.seed_everything(args.seed, workers=True)
    

    # os.environ['WORLD_SIZE'] = '8'

    # print('MASTER_ADDR', os.environ['MASTER_ADDR'])
    # print('MASTER_PORT', os.environ['MASTER_PORT'])
    # print('LOCAL_RANK', os.environ['LOCAL_RANK'])
    # print('WORLD_SIZE', os.environ['WORLD_SIZE'])
    # exit()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    ### Define datasets
    if args.exp == 'elevation' and args.mode == 'inference':
        from data import create_elevation_inference_dataloader
        dataloader = create_elevation_inference_dataloader(config)
    else:
        dataloader = create_dataloader(config)
    if args.mode != 'inference':
        val_dataloader = create_val_dataloader(config)
    ### Define model
    cpu_opt = config.cpu_offload
    if args.mode == 'inference':
        cpu_opt = True
    if args.exp == 'diffusion':
        trainer_model = StableDiffusionTrainer(config.ddconfig)
    elif args.exp == 'renderer':
        trainer_model = RendererTrainer(config.renderconfig)
    elif args.exp == 'dit':
        trainer_model = STDITTrainer(config.ddconfig)
    elif args.exp == 'flux':
        from trainer.trainer_flux import Flux_Trainer
        trainer_model = Flux_Trainer(config=config,init_step=args.init_step)
    elif args.exp == 'pose':
        from trainer.trainer_pose2 import Pose_Trainer
        trainer_model = Pose_Trainer(config=config,init_step=args.init_step)
    elif args.exp == 'trellis':
        from trainer.trainer_trellis import TrellisPoseDecoupledTrainer
        trainer_model = TrellisPoseDecoupledTrainer(config=config)
    elif args.exp == 'rotator':
        from trainer.trainer_rotator import Trainer_Rotator
        trainer_model = Trainer_Rotator(config=config)
    elif args.exp == 'rotator2':#扭转器
        from trainer.trainer_rotator2 import Trainer_Rotator
        trainer_model = Trainer_Rotator(config=config)
    elif args.exp == 'elevation':#角度估计器
        from trainer.trainer_elevation import Trainer_Elevation
        trainer_model = Trainer_Elevation(config=config)
    elif args.exp == 'trellis_pipeline':
        from trainer.trainer_trellis_pipeline import Trainer_TrellisPipeline
        trainer_model = Trainer_TrellisPipeline(config=config,val_use_full_sparse_sampling=True)

    ### Define trainer


    
    checkpoint_callback = ModelCheckpoint(
        dirpath                   =     os.path.join(config.savedir, 'checkpoints'),
        filename                  =     config.name + '-{step}', # -{epoch:02d}
        monitor                   =     'step',
        save_last                 =     True,
        save_top_k                =     -1,
        verbose                   =     True,
        every_n_train_steps       =     args.every_n_train_steps,
        save_on_train_epoch_end   =     False,
    )

    strategy = DeepSpeedStrategy(
        stage                     =     2 if cpu_opt else 1, 
        offload_optimizer         =     cpu_opt, 
        overlap_comm=False,
        
        #logging_level=logging.DEBUG
        # offload_parameters        =     True,
        # offload_params_device     =     'cpu',
        # cpu_checkpointing         =     True,
    )
    image_logger = ImageLogger(50)
    #reset_callback = ResetOptimizerCallback()
    signal_callback = CheckpointOnInterrupt(save_path=os.path.join(config.savedir, 'safetensors/-1/'))
    trainer = pl.Trainer(
        default_root_dir          =     config.savedir,
        callbacks                 =     [checkpoint_callback, lr_monitor, ModelSummary(4),signal_callback], # ModelSummary(2)
        accelerator               =     'gpu',
        accumulate_grad_batches   =     args.grad_acc ,
        benchmark                 =     True,
        num_nodes                 =     args.nodes,
        devices                   =     args.devices,
        gradient_clip_val         =     config.max_grad_norm,
        log_every_n_steps         =     2,
        precision                 =     'bf16-mixed', #"bf16",
        max_epochs                =     config.num_train_epochs,
        strategy                  =     strategy,
        sync_batchnorm            =     True,
        #max_time                  =     "00:08:00:00",
        val_check_interval        =     100*config.gradient_accumulation_steps,
        limit_val_batches         =     50,
        check_val_every_n_epoch   =     None,
        plugins                   =     [AsyncCheckpointIO()],
    )
    if args.resume != '':
        print(f" resuming from {args.resume}")
        assert os.path.exists(args.resume), "resume path does not exist"
    if args.mode == 'train':
        ### training
        trainer.fit(
            model                     =     trainer_model,
            train_dataloaders         =     dataloader,
            val_dataloaders           =     val_dataloader,
            ckpt_path                 =     None if not os.path.exists(args.resume) else args.resume,
            
        )
    elif args.mode == 'validate':
        assert os.path.exists(args.resume), "resume path does not exist"
        trainer.validate(
            model                     =     trainer_model,
            dataloaders               =     val_dataloader,
            ckpt_path                 =     args.resume,
        )
    elif args.mode == 'test':
        assert os.path.exists(args.resume), "resume path does not exist"
        trainer.test(
            model                     =     trainer_model,
            dataloaders               =     dataloader,
            ckpt_path                 =     args.resume,
        )
    elif args.mode == 'inference':
        ckpt_path = args.resume if args.resume else None
        if args.exp == 'elevation':
            # elevation 推理：直接循环推理 + 写 JSON，绕过 PL predict（无需 ckpt）
            import json
            import tempfile
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            device = torch.device(f"cuda:{local_rank}")
            trainer_model.eval()
            trainer_model.to(device)

            rank_results = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    result = trainer_model.predict_step(batch, -1)
                    rank_results.extend(result)

            output_path = trainer_model.inference_output_path
            # 各 rank 写临时文件，rank 0 合并
            tmp_dir = os.path.dirname(output_path) or tempfile.gettempdir()
            rank_file = os.path.join(tmp_dir, f".elevation_rank_{local_rank}.json")
            with open(rank_file, 'w') as f:
                json.dump(rank_results, f)
            print(f"[Rank {local_rank}] {len(rank_results)} samples -> {rank_file}")

            # 等待所有 rank 写完
            if world_size > 1:
                torch.cuda.synchronize()
                # barrier via file: rank 0 waits for all rank files
                if local_rank == 0:
                    for r in range(world_size):
                        rf = os.path.join(tmp_dir, f".elevation_rank_{r}.json")
                        while not os.path.exists(rf):
                            time.sleep(0.1)
                    # 合并
                    all_results = []
                    for r in range(world_size):
                        rf = os.path.join(tmp_dir, f".elevation_rank_{r}.json")
                        with open(rf, 'r') as f:
                            all_results.extend(json.load(f))
                        os.remove(rf)
                else:
                    # non-rank-0: wait for rank 0 to clean up
                    while os.path.exists(rank_file):
                        time.sleep(0.2)
            else:
                all_results = rank_results

            if local_rank == 0:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"[Inference] {len(all_results)} samples saved to {output_path}")
        else:
            with torch.no_grad():
                trainer.predict(
                    model                     =     trainer_model,
                    dataloaders                =     dataloader,
                    ckpt_path                 =     ckpt_path,
                )

if __name__== '__main__':
    main_func()
