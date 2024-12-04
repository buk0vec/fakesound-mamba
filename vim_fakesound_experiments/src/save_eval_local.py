import torch
from models import LitViM
from datasets import ReconstructedFakeSoundDatasetS3
import lightning as L
import argparse
from transforms import transform
from stats import compute_stats
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--input_bucket", type=str, default="")
    parser.add_argument("--model_name", type=str, default="model")
    
    num_workers = 0 # using s3 dataset, not fork safe
    batch_size = 8

    args = vars(parser.parse_args())

    model = LitViM.load_from_checkpoint(checkpoint_path=args['model_ckpt'])

    easy_set = ReconstructedFakeSoundDatasetS3(args['input_bucket'], split='easy', transform=transform)
    hard_set = ReconstructedFakeSoundDatasetS3(args['input_bucket'], split='hard', transform=transform)
    zeroshot_set = ReconstructedFakeSoundDatasetS3(args['input_bucket'], split='zeroshot', transform=transform)

    easy_loader = torch.utils.data.DataLoader(easy_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    hard_loader = torch.utils.data.DataLoader(hard_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    zeroshot_loader = torch.utils.data.DataLoader(zeroshot_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    trainer = L.Trainer()
    loaders = [easy_loader, hard_loader, zeroshot_loader]
    names = ['easy', 'hard', 'zeroshot']
    # loaders = [easy_loader, hard_loader]
    # names = ['easy', 'hard']
    predictions = trainer.predict(model, loaders)

    # resolutions = [None, 2, 100]
    print(len(predictions))

    for i in range(len(loaders)):
        preds = predictions[i]
        pred_class_batches = np.concatenate([p[0].numpy() for p in preds], axis=0) # n x 1
        pred_seg_batches = np.concatenate([p[1].numpy() for p in preds], axis=0) # n x t
        print(pred_class_batches.shape)
        print(pred_seg_batches.shape)
        class_batches = np.zeros(pred_class_batches.shape)
        seg_batches = np.zeros(pred_seg_batches.shape)
        counter = 0
        for idx, batch in enumerate(loaders[i]):
            _, label, seg = batch
            B = label.shape[0]
            class_batches[counter:counter+B] = label[:, np.newaxis]
            seg_batches[counter:counter+B] = seg
            counter += B
        # classes = np.stack([pred_class_batches, class_batches], axis=0)
        # segs = np.stack([pred_seg_batches, seg_batches], axis=0)
        np.savetxt(f"../stats/{args['model_name']}_{names[i]}_class_pred.csv", pred_class_batches, delimiter=",", fmt="%.8f")
        np.savetxt(f"../stats/{args['model_name']}_{names[i]}_class.csv", class_batches, delimiter=",", fmt="%.8f")
        np.savetxt(f"../stats/{args['model_name']}_{names[i]}_seg_pred.csv", pred_seg_batches, delimiter=",", fmt="%.8f")
        np.savetxt(f"../stats/{args['model_name']}_{names[i]}_seg.csv", seg_batches, delimiter=",", fmt="%.8f")
        # stats = [[], [], []]
    # for i in range(len(loaders)):
    #     class_preds = np.zeros(len(loaders[i].dataset), 2)
    #     loader = loaders[i]
    #     name = names[i]
    #     stats_acc = [{
    #         'class_acc': [],
    #         'f1': [],
    #         'score': [],
    #         'prec': [],
    #         'recall': [],
    #         'n': []
    #     } for _ in range(len(resolutions))]

    #     for idx, batch in enumerate(loader):
    #         _, labels, segments = batch
    #         B = labels.shape[0]
    #         labels_, segments_ = predictions[i][idx]
    #         for r in len(resolutions): 
    #             score, class_acc, seg_prec, seg_recall, seg_f1 = compute_stats(labels_, segments_, labels, segments, scale=resolutions[r])
    #             stats_acc[r]['class_acc'].append(class_acc)
    #             stats_acc[r]['f1'].append(seg_f1)
    #             stats_acc[r]['score'].append(score)
    #             stats_acc[r]['prec'].append(seg_prec)
    #             stats_acc[r]['recall'].append(seg_recall)
    #             stats_acc[r]['n'].append(labels.shape[0])
    #     for r in len(resolutions):
    #         stats_acc[r] = {k: np.array(stats_acc[r][k]) for k in stats_acc[r]}
    #         stats_agg = {k: np.sum(stats_acc[r][k] * stats_acc[r]['n']) / np.sum(stats_acc[r]['n']) for k in stats_acc[r]}
    #         del stats_agg['n']
    #         stats_agg['resolution'] = resolutions[i]
    #         stats[i].append(stats_agg)

    # for i in range(3):
    #     print(f"\n{name[i]} set")
    #     for r in range(3):
    #         print(f"Resolution {resolutions[r]}")
    #         print(stats[i][r])
        


