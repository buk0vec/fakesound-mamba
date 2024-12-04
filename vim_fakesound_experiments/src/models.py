import torch
from torch import nn
import torchaudio.transforms as T
from Vim.vim.models_mamba import VisionMamba
import lightning as L
from stats import compute_stats
from torchvision.ops import sigmoid_focal_loss

class LitViM(L.LightningModule):
    def __init__(
        self, 
        embed_dim=384, 
        depth=24,
        learning_rate=1e-4, 
        weight_decay=0.05, 
        warmup_epochs=20, 
        warmup_lr=1e-5,
        cosine_t=20,
        cosine_min=1e-5,
        cosine_epochs=260,
        cooldown_lr = 1e-5,
        cooldown_epochs=40,
        patch_size=(140, 4),
        stride=(140, 4),
        img_size=(280,1000),
        num_classes=1,
        loss_alpha=0.3,
        bce_weight=1,
        focal_alpha=0.25,
        drop_path_rate=0.05,
        drop_rate=0,
        seq_pool_type='mean',
        seq_patch_transform='stack',
        if_seq_mamba=False,
        seq_mamba_depth=8,
        if_seq_residual=False,
        seq_loss='focal'
    ):
        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()
        self.vim = VisionMamba(
            patch_size=patch_size, 
            stride=stride, 
            img_size=img_size, 
            num_classes=num_classes, 
            depth=depth, 
            channels=1, 
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            if_sequence_task=True,
            remove_cls_token=True,
            embed_dim=embed_dim,
            seq_pool_type=seq_pool_type,
            seq_patch_transform=seq_patch_transform,
            if_seq_mamba=if_seq_mamba,
            seq_mamba_depth=seq_mamba_depth,
            if_seq_residual=if_seq_residual
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs=warmup_epochs
        self.warmup_lr = warmup_lr
        self.cosine_epochs = cosine_epochs
        self.cosine_t = cosine_t
        self.cosine_min = cosine_min
        self.cooldown_epochs = cooldown_epochs
        self.cooldown_lr = cooldown_lr
        self.loss_alpha = loss_alpha
        # self.bce_weight = bce_weight
        self.focal_alpha = focal_alpha
        self.seq_loss = seq_loss

        self.register_buffer("bce_weight", torch.tensor([bce_weight]))
        
    def training_step(self, batch, batch_idx):
        inputs, labels, segments = batch
        # opt = self.optimizers()
        # opt.zero_grad()
        # print("Getting data")
        # print(f"Passing data, {inputs[0].shape}")
        labels_, segments_ = self.vim(inputs)
        # print(f"Class: {labels_.view(-1)} {labels_.view(-1).shape}, {labels} {labels.shape}")
        class_loss = nn.functional.binary_cross_entropy_with_logits(labels_.view(-1), labels)
        if self.seq_loss == 'focal':
            segment_loss = sigmoid_focal_loss(segments_, segments, alpha=self.focal_alpha, reduction='mean')
        else:
            segment_loss = nn.functional.binary_cross_entropy_with_logits(segments_, segments, pos_weight=self.bce_weight)
        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * segment_loss
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('train_class_loss', class_loss, on_epoch=True, sync_dist=True)
        self.log('train_seg_loss', segment_loss, on_epoch=True, sync_dist=True)
        return loss
        # self.manual_backward(loss)
        # opt.step()
        # seg_loss = nn.functional.binary_cross_entropy(segments_, segments)
        # print(f"Seg shapes: {segments_.shape}, {segments.shape}")
        # self.log('train_loss', loss, on_epoch=True)

    # def on_train_epoch_end(self):
    #     sch = self.lr_schedulers()
    #     sch.step()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _, _, = batch
        labels_, segments_ = self.vim(inputs)
        labels_ = torch.sigmoid(labels_)
        segments_ = torch.sigmoid(segments_)
        return labels_, segments_
        
    def validation_step(self, batch, batch_idx):
        inputs, labels, segments = batch
        labels_, segments_ = self.vim(inputs)
        class_loss = nn.functional.binary_cross_entropy_with_logits(labels_.view(-1), labels)
        if self.seq_loss == 'focal':
            segment_loss = sigmoid_focal_loss(segments_, segments, alpha=self.focal_alpha, reduction='mean')
        else:
            segment_loss = nn.functional.binary_cross_entropy_with_logits(segments_, segments, pos_weight=self.bce_weight)
        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * segment_loss        
        self.log("val_loss", torch.mean(loss), on_epoch=True, sync_dist=True)
        labels_ = torch.sigmoid(labels_)
        segments_ = torch.sigmoid(segments_)
        score, class_acc, seg_prec, seg_recall, seg_f1 = compute_stats(labels_, segments_, labels, segments)
        self.log("val_class_loss", class_loss, on_epoch=True, sync_dist=True)
        self.log("val_seg_loss", segment_loss, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_class_acc", class_acc, on_epoch=True, sync_dist=True)
        self.log("val_seg_prec", seg_prec, on_epoch=True, sync_dist=True)
        self.log("val_seg_recall", seg_recall, on_epoch=True, sync_dist=True)
        self.log("val_seg_f1", seg_f1, on_epoch=True, sync_dist=True)
        self.log("val_score", score, on_epoch=True, sync_dist=True)
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # warmup = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.01, total_iters=20)
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 + (((epoch/self.warmup_epochs * self.learning_rate + (self.warmup_epochs - epoch)/self.warmup_epochs * self.warmup_lr ) - self.learning_rate) / self.learning_rate))
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.cosine_t, eta_min=self.cosine_min)
        # cooldown = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - ((self.learning_rate - self.cooldown_lr) / self.learning_rate) * (epoch / (self.cooldown_epochs - 1)))
        cooldown = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.cooldown_lr / self.learning_rate)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine, cooldown], milestones=[self.warmup_epochs, self.warmup_epochs + self.cosine_epochs])
        # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]