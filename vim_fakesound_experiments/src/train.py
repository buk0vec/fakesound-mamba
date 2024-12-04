import torch
from torch import nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

import os
import argparse
import json
import glob
import re

from models import LitViM
from transforms import transform, augment_spec
from datasets import ReconstructedFakeSoundDataset
from ema import EMA
from utils import from_vim_pretrained

pretrained = "vim_s_midclstok_80p5acc.pth"

def epoch_from_filename(filename):
    match = re.search(r'epoch=(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def locate_checkpoint(checkpoint_dir, model_name):
    print(f"Locating checkpoint from {checkpoint_dir}...")
    if not os.path.exists(checkpoint_dir):
        print(f"Couldn't locate {checkpoint_dir}")
        return None
    files = glob.glob(f"{checkpoint_dir}/**/*.ckpt", recursive=True)
    chkpts = [f for f in files if os.path.basename(f).startswith(model_name + '-epoch')]
    if len(chkpts) == 0:
        print(f"No PL checkpoint for {model_name} found in checkpoint dir")
        files = glob.glob(f"{checkpoint_dir}/**/{pretrained}", recursive=True)
        return files[0] if len(files) > 0 else None
    print(f"Locating latest checkpoint from {chkpts}")
    return max(chkpts, key=epoch_from_filename)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--loss_alpha", type=float, default=0.3)
    parser.add_argument("--bce_weight", type=float, default=1)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2)
    parser.add_argument("--patch_width", type=int, default=4)
    parser.add_argument("--patch_height", type=int, default=140)
    parser.add_argument("--if_augment", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--if_sm_tuning", type=int, default=0)

    parser.add_argument("--spec_width", type=int, default=1000)
    parser.add_argument("--spec_height", type=int, default=280)
    parser.add_argument("--seq_pool_type", type=str, default='max')
    parser.add_argument("--seq_patch_transform", type=str, default='stack')
    parser.add_argument("--if_seq_mamba", type=int, default=0)
    parser.add_argument("--seq_mamba_depth", type=int, default=12)
    parser.add_argument("--if_seq_residual", type=int, default=0)
    parser.add_argument("--seq_loss", type=str, default='focal')
    
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--drop_path", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_val_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--pin_val_memory", type=int, default=0)
    parser.add_argument("--persist_workers", type=int, default=0)
    parser.add_argument("--persist_val_workers", type=int, default=0)
    parser.add_argument("--every_n_epochs", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--cosine_epochs", type=int, default=300)
    parser.add_argument("--cosine_t", type=int, default=300)
    parser.add_argument("--cosine_min", type=float, default=1e-5)
    parser.add_argument("--cooldown_epochs", type=int, default=100)
    parser.add_argument("--cooldown_lr", type=float, default=1e-5)
    parser.add_argument("--ema", type=int, default=0)
    parser.add_argument("--val_batches", type=int, default=0)
    
    parser.add_argument("--model_name", type=str, default="model")
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints")
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = vars(parser.parse_args())

    bool_keys = ['pin_memory', 'pin_val_memory', 'persist_workers', 'persist_val_workers', 'if_seq_mamba', 'if_seq_residual', 'if_augment', 'if_sm_tuning']
    for key in bool_keys:
        args[key] = (args[key] == 1)
        print(f"Set {key} to {args[key]}")

    # print(os.listdir(args['data_dir']))
    epochs = args['warmup_epochs'] + args['cosine_epochs'] + args['cooldown_epochs']

    patch_size = (args['patch_height'], args['patch_width'])
    stride = patch_size
    img_size = (args['spec_height'], args['spec_width'])

    augment = augment_spec if args['if_augment'] else None

    if args['if_sm_tuning']:
        args['model_name'] = args['model_name'] + f"_{args['bce_weight']}_{args['dropout']}_{args['drop_path']}_{args['weight_decay']}"
        print(f"Renamed model to ${args['model_name']} for SM HPO")

    train_set = ReconstructedFakeSoundDataset(root_dir=args['data_dir'], split='train', transform=transform, augment=augment)
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [1 - args['val_split'], args['val_split']], 
        # for evaluating model architectures fairly
        generator= torch.Generator().manual_seed(42) 
    )
    val_set.dataset.augment = None
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"], num_workers=args["num_workers"], pin_memory=args["pin_memory"], persistent_workers=args['persist_workers'], drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args["batch_size"], num_workers=args["num_val_workers"], pin_memory=["pin_val_memory"], persistent_workers=args['persist_val_workers'], drop_last=False)

    chkpt_callback = L.pytorch.callbacks.ModelCheckpoint(save_top_k=1, dirpath=args["checkpoint_path"], filename=args['model_name']+'-{epoch:02d}-{train_loss:.2f}-{val_score:.3f}')
    
    checkpoint = locate_checkpoint(args["checkpoint_path"], args["model_name"])
    # we don't do this because apparently you do this in trainer.fit()
    # if checkpoint and checkpoint.endswith('.ckpt'): # if pl checkpoint
    #     print(f"Attempting to load from checkpoint {checkpoint}")
    #     model = LitViM.load_from_checkpoint(checkpoint_path=checkpoint)

    if checkpoint and checkpoint.endswith('.pth'): # if vim checkpoint
        print(f"Loading from pretrained ViM checkpoint {checkpoint}")
        model = from_vim_pretrained(checkpoint, **args)
    else:
        if checkpoint and checkpoint.endswith('.ckpt'):
            print(f"Found checkpoint {checkpoint} and will use for training")
        else:
            print("No checkpoint found, reinit model")
        model = LitViM(
            embed_dim=args["embed_dim"],
            depth=args["depth"], 
            learning_rate=args["learning_rate"], 
            weight_decay=args["weight_decay"], 
            warmup_lr=args["warmup_lr"],
            warmup_epochs=args["warmup_epochs"],
            cosine_t=args["cosine_t"],
            cosine_min=args["cosine_min"],
            cosine_epochs=args["cosine_epochs"],
            cooldown_lr=args["cooldown_lr"],
            cooldown_epochs=args["cooldown_epochs"],
            loss_alpha=args['loss_alpha'],
            bce_weight=args['bce_weight'],
            focal_alpha=args['focal_alpha'],
            patch_size=patch_size,
            stride=stride,
            img_size=img_size,
            seq_pool_type=args['seq_pool_type'],
            seq_loss=args['seq_loss'],
            drop_path_rate=args['drop_path'],
            drop_rate=args['dropout'],
            seq_patch_transform=args['seq_patch_transform'],
            if_seq_mamba=args['if_seq_mamba'],
            seq_mamba_depth=args['seq_mamba_depth'],
            if_seq_residual=args['if_seq_residual']
        )

    torch.cuda.empty_cache()

    logger = TensorBoardLogger(save_dir=f"s3://bukovec-ml-data/logs/{args['model_name']}")
    device_stats = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if args['ema'] == 1:
        callbacks = [chkpt_callback, lr_monitor, device_stats, EMA(0.9999)]
    else:
        callbacks = [chkpt_callback, lr_monitor, device_stats]
    if args['num_gpus'] > 1:
        trainer = L.Trainer(accelerator="gpu",
                            devices=args["num_gpus"],
                            strategy=DDPStrategy(gradient_as_bucket_view=True),
                            max_epochs=epochs,
                            callbacks=callbacks,
                            logger=logger,
                            check_val_every_n_epoch=args["every_n_epochs"],
                            limit_val_batches=args['val_batches'],
                            enable_progress_bar=False
                           )
    else:
        trainer = L.Trainer(accelerator="gpu", 
                            max_epochs=epochs, 
                            callbacks=callbacks, 
                            logger=logger, 
                            check_val_every_n_epoch=args["every_n_epochs"],
                            limit_val_batches=args['val_batches'],
                            enable_progress_bar=False
                           )

    if checkpoint and checkpoint.endswith('.ckpt'):
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(f"{args['model_dir']}/{args['model_name']}.ckpt")

    print("Running validation with final model")

    trainer = L.Trainer(enable_progress_bar=False)
    
    val_metrics = trainer.validate(model, val_loader)[0]

    for metric in val_metrics:
        print(f"final_{metric} = {val_metrics[metric]}")

    # trainer.test(model, test_loader)