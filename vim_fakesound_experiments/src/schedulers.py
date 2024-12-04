import torch

def create_cosine_scheduler(optimizer, learning_rate, warmup_lr, warmup_epochs, cosine_min, cosine_epochs, cooldown_lr, cooldown_epochs):
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 + (((epoch/warmup_epochs * learning_rate + (warmup_epochs - epoch)/warmup_epochs * warmup_lr ) - learning_rate) / learning_rate))
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cosine_epochs, eta_min=cosine_min)
    cooldown = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: cooldown_lr / learning_rate)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine, cooldown], milestones=[warmup_epochs, warmup_epochs + cosine_epochs])
    return scheduler