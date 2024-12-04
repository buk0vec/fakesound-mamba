import torch

def segmentation_metrics(pred, target):
    tp = (pred * target).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()

    return tp, fp, fn

# scale a 10ms resolution tensor down
# for 20ms, scale = 2. for 1s, scale=100
def scale_batch_tensor(tensor, scale):
    tensor = tensor.view(tensor.shape[0], -1, scale) # b x (s/c) x c
    tensor = torch.max(tensor, dim=2)[0]
    return tensor

def compute_stats(vclass_, vseg_, vclass, vseg, scale=None, verbose=False):
    # easy_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    class_correct = 0
    n_samples = 0
    seg_tp = 0
    seg_fp = 0 
    seg_fn = 0
    alpha = 0.3
    epsilon = 1e-8

    vclass_ = (vclass_.view(-1) >= 0.5).float()
    vseg_ = (vseg_ >= 0.5).float()
    
    if scale:
        vseg = scale_batch_tensor(vseg, scale)
        vseg_ = scale_batch_tensor(vseg_, scale)

    class_correct += (vclass_ == vclass).sum()
    n_samples += vclass_.shape[0]
    tp, fp, fn = segmentation_metrics(vseg_, vseg)
    seg_tp += tp
    seg_fp += fp
    seg_fn += fn

    class_acc = class_correct/n_samples
    seg_prec = seg_tp / (seg_tp + seg_fp + epsilon)
    seg_recall = seg_tp / (seg_tp + seg_fn + epsilon)
    seg_f1 = 2 / ((1 / (seg_prec + epsilon)) + (1/(seg_recall + epsilon)))
    score = alpha * class_acc + (1-alpha) * seg_f1
    if verbose:
        if scale:
            print(f"Scale factor: {scale}")
        print(f"Class Accuracy = {class_correct} / {n_samples} = {class_acc.item()}, Segment Precision = {seg_prec}, Segment Recall = {seg_recall}. Segment F1 = {seg_f1}")
        print(f"Score = {score}")

    return score, class_acc, seg_prec, seg_recall, seg_f1
