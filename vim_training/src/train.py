from datasets import ImagenetteDataset
import torch
import torchvision
import lightning as L
import os
from torch import nn
from torchvision import transforms
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import json
import glob


from Vim.vim.models_mamba import VisionMamba
from ema import EMA

logger = TensorBoardLogger(save_dir="s3://bukovec-ml-data/logs/vit-imagenette-small")

transform = transforms.Compose([
    # torchvision.transforms.Resize((384, 384)),
    transforms.RandomResizedCrop((384, 384), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LitViM(L.LightningModule):
    def __init__(
        self, 
        depth=16, 
        learning_rate=1e-4, 
        weight_decay=0.05, 
        warmup_epochs=20, 
        warmup_lr=1e-5,
        cosine_t=20,
        cosine_min=1e-5,
        cosine_epochs=260,
        cooldown_lr = 1e-5,
        cooldown_epochs=40
    ):
        super().__init__()
        self.automatic_optimization = False
        self.vim = VisionMamba(patch_size=24, stride=24, img_size=384, num_classes=10, depth=depth, channel=3, drop_path_rate=0)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs=warmup_epochs
        self.warmup_lr = warmup_lr
        self.cosine_epochs = cosine_epochs
        self.cosine_t = cosine_t
        self.cosine_min = cosine_min
        self.cooldown_epochs = cooldown_epochs
        self.cooldown_lr = cooldown_lr
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        opt = self.optimizers()
        opt.zero_grad()
        # print("Getting data")
        # print(f"Passing data, {inputs[0].shape}")
        labels_ = self.vim(inputs)
        # print(f"Class: {labels_.view(-1)} {labels_.view(-1).shape}, {labels} {labels.shape}")
        loss = nn.functional.cross_entropy(labels_, labels)
        self.manual_backward(loss)
        opt.step()
        # seg_loss = nn.functional.binary_cross_entropy(segments_, segments)
        # print(f"Seg shapes: {segments_.shape}, {segments.shape}")
        self.log('train_loss', loss, on_epoch=True)
        # return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()
            
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_= self.vim(inputs)
        loss = nn.functional.cross_entropy(labels_, labels)
        labels_ = torch.argmax(labels_, dim=1)
        labels = torch.argmax(labels, dim=1)
        accuracy = torch.mean((labels == labels_).float())
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True) 

    

            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # warmup = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.01, total_iters=20)
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 + (((epoch/self.warmup_epochs * self.learning_rate + (self.warmup_epochs - epoch)/self.warmup_epochs * self.warmup_lr ) - self.learning_rate) / self.learning_rate))
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.cosine_t, eta_min=self.cosine_min)
        # cooldown = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - ((self.learning_rate - self.cooldown_lr) / self.learning_rate) * (epoch / (self.cooldown_epochs - 1)))
        cooldown = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.cooldown_lr / self.learning_rate)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine, cooldown], milestones=[self.warmup_epochs, self.warmup_epochs + self.cosine_epochs])
        # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def locate_checkpoint(checkpoint_dir):
    print(f"Locating checkpoint from {checkpoint_dir}...")
    if not os.path.exists(checkpoint_dir):
        print(f"Couldn't locate {checkpoint_dir}")
        return None
    files = glob.glob(f"{checkpoint_dir}/**/*.ckpt", recursive=True)
    if len(files) == 0:
        print("No files found in checkpoint dir")
        return None
    print(f"Found files: {files}")
    return files[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--depth", type=int, default=24)
    # parser.add_argument("--n_blocks", type=int, default=12)
    # parser.add_argument("--loss_alpha", type=float, default=0.3)
    # parser.add_argument("--focal_alpha", type=float, default=0.55)
    # parser.add_argument("--focal_gamma", type=float, default=2)
    
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_val_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--pin_val_memory", type=int, default=0)
    parser.add_argument("--persist_workers", type=int, default=0)
    parser.add_argument("--persist_val_workers", type=int, default=0)
    parser.add_argument("--every_n_epochs", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--cosine_epochs", type=int, default=260)
    parser.add_argument("--cosine_t", type=int, default=20)
    parser.add_argument("--cosine_min", type=float, default=1e-5)
    parser.add_argument("--cooldown_epochs", type=int, default=20)
    parser.add_argument("--cooldown_lr", type=float, default=1e-5)
    parser.add_argument("--ema", type=int, default=1)
    parser.add_argument("--val_batches", type=int, default=0)
    
    parser.add_argument("--model_name", type=str, default="model")
    
    
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints")
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = vars(parser.parse_args())

    print(os.listdir(args['data_dir']))
    epochs = args['warmup_epochs'] + args['cosine_epochs'] + args['cooldown_epochs']

    train_set = ImagenetteDataset(root_dir=args['data_dir'], split='train', transform=transform)
    val_set = ImagenetteDataset(root_dir=args['data_dir'], split='val', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"], num_workers=args["num_workers"], pin_memory=(args["pin_memory"] == 1), persistent_workers=(args['persist_workers'] == 1), drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args["batch_size"], num_workers=args["num_val_workers"], pin_memory=(args["pin_val_memory"] == 1), persistent_workers=(args['persist_val_workers'] == 1), drop_last=False)

    chkpt_callback = L.pytorch.callbacks.ModelCheckpoint(save_top_k=1, monitor="epoch", mode='max', dirpath=args["checkpoint_path"], filename='vit-imagenette-small-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}', every_n_epochs=args["every_n_epochs"])
    
    checkpoint = locate_checkpoint(args["checkpoint_path"])
    if checkpoint:
        model = LitViM.load_from_checkpoint(checkpoint_path=f"{args['checkpoint_path']}/{checkpoint}")
    else:
        model = LitViM(depth=args["depth"], learning_rate=args["learning_rate"], weight_decay=args["weight_decay"], warmup_lr=args["warmup_lr"],warmup_epochs=args["warmup_epochs"],cosine_t=args["cosine_t"],cosine_min=args["cosine_min"],cosine_epochs=args["cosine_epochs"],cooldown_lr=args["cooldown_lr"],cooldown_epochs=args["cooldown_epochs"])
    model.train()
    if args['ema'] == 1:
        callbacks = [chkpt_callback, EMA(0.9999)]
    else:
        callbacks = [chkpt_callback]
    trainer = L.Trainer(accelerator="gpu", 
                        max_epochs=epochs, 
                        callbacks=callbacks, 
                        logger=logger, 
                        check_val_every_n_epoch=args["every_n_epochs"],
                        limit_val_batches=args['val_batches']
                       )
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(f"{args['model_dir']}/{args['model_name']}.ckpt")

    # trainer.test(model, test_loader)
    
    

    