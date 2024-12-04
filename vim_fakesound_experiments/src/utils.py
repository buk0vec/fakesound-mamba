from models import LitViM
import torch
import torch.nn.functional as F
import lightning as L


def from_vim_pretrained(path, **kwargs):
    
    img_size = (kwargs['spec_height'], kwargs['spec_width'])
    patch_size = (kwargs['patch_height'], kwargs['patch_width'])
    stride = patch_size # no overlap
    
    model = LitViM(
        embed_dim=kwargs["embed_dim"],
        depth=kwargs["depth"], 
        learning_rate=kwargs["learning_rate"], 
        weight_decay=kwargs["weight_decay"], 
        warmup_lr=kwargs["warmup_lr"],
        warmup_epochs=kwargs["warmup_epochs"],
        cosine_t=kwargs["cosine_t"],
        cosine_min=kwargs["cosine_min"],
        cosine_epochs=kwargs["cosine_epochs"],
        cooldown_lr=kwargs["cooldown_lr"],
        cooldown_epochs=kwargs["cooldown_epochs"],
        loss_alpha=kwargs['loss_alpha'],
        bce_weight=kwargs['bce_weight'],
        focal_alpha=kwargs['focal_alpha'],
        patch_size=patch_size,
        num_classes=1,
        stride=stride,
        img_size=img_size,
        seq_pool_type=kwargs['seq_pool_type'],
        seq_loss=kwargs['seq_loss'],
        drop_path_rate=kwargs['drop_path'],
        drop_rate=kwargs['dropout'],
        seq_patch_transform=kwargs['seq_patch_transform'],
        if_seq_mamba=kwargs['if_seq_mamba'],
        seq_mamba_depth=kwargs['seq_mamba_depth'],
        if_seq_residual=kwargs['if_seq_residual']
    )
    
    sd = torch.load(path, weights_only=False, map_location='cpu')
    sd = sd['model']
    # drop linear classifier weights and bias
    for k in ['head.weight', 'head.bias']:
        del sd[k]
    
    # modified from https://github.com/hustvl/Vim/blob/main/vim/main.py
    # interpolate position embedding, 
    pos_embed_checkpoint = sd['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.vim.patch_embed.num_patches
    num_extra_tokens = model.vim.pos_embed.shape[-2] - num_patches
    
    # height (== width) for the checkpoint position embedding
    orig_height = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    orig_width = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height and width for the new position embedding
    new_height = int(img_size[0] / model.vim.patch_embed.patch_size[0])
    new_width = int(img_size[1] / model.vim.patch_embed.patch_size[1])
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_height, orig_width, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_height, new_width), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    sd['pos_embed'] = new_pos_embed
    
    # interpolate patch embeddings, idk how effective this is vs reinitializing but i don't think it hurts
    W_pe = sd['patch_embed.proj.weight']
    W_pe = F.interpolate(W_pe, size=patch_size, mode='bilinear', align_corners=True)
    W_pe = torch.mean(W_pe, axis=1, keepdims=True)
    sd['patch_embed.proj.weight'] = W_pe
    
    model.vim.load_state_dict(sd, strict=False)

    del sd
    
    return model