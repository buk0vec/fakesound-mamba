import sagemaker
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter
from sagemaker.estimator import Estimator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--loss_alpha", type=float, default=0.3)
    # parser.add_argument("--bce_weight", type=float, default=1)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2)
    parser.add_argument("--patch_width", type=int, default=10)
    parser.add_argument("--patch_height", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--if_sm_tuning", type=int, default=1) # tuning!


    parser.add_argument("--spec_width", type=int, default=1000)
    parser.add_argument("--spec_height", type=int, default=280)
    parser.add_argument("--seq_pool_type", type=str, default='mean')
    parser.add_argument("--seq_patch_transform", type=str, default='stack')
    parser.add_argument("--if_seq_mamba", type=int, default=0)
    parser.add_argument("--seq_mamba_depth", type=int, default=12)
    parser.add_argument("--if_seq_residual", type=int, default=0)
    parser.add_argument("--seq_loss", type=str, default='focal')
    parser.add_argument("--if_augment", type=int, default=0)


    parser.add_argument("--learning_rate", type=float, default=5e-4)
    # parser.add_argument("--weight_decay", type=float, default=0.01)
    # parser.add_argument("--drop_path", type=float, default=0.05)
    # parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--num_val_workers", type=int, default=12)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--pin_val_memory", type=int, default=1)
    parser.add_argument("--persist_workers", type=int, default=0)
    parser.add_argument("--persist_val_workers", type=int, default=0)
    parser.add_argument("--every_n_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--cosine_epochs", type=int, default=200) # only 200 cosine epochs for tuning
    parser.add_argument("--cosine_t", type=int, default=300)
    parser.add_argument("--cosine_min", type=float, default=1e-5)
    parser.add_argument("--cooldown_epochs", type=int, default=50)
    parser.add_argument("--cooldown_lr", type=float, default=1e-5)
    parser.add_argument("--ema", type=int, default=0)
    # can be larger than number of actual batches. also setting to 0 disables checkpointing, just be aware
    parser.add_argument("--val_batches", type=int, default=1000) 
    parser.add_argument("--model_name", type=str, default="model")

    parser.add_argument("--instance_type", type=str, default="ml.g5.48xlarge")
    parser.add_argument("--input_bucket", type=str, default="")
    parser.add_argument("--image_uri", type=str, default="")
    parser.add_argument("--output_bucket", type=str, default="")
    parser.add_argument("--checkpoint_bucket", type=str, default="")

    args = vars(parser.parse_args())
    hyperparameters = args.copy()
    instance_type = args['instance_type']
    for k in [
        'instance_type', 
        'input_bucket', 
        'image_uri', 
        'output_bucket', 
        'checkpoint_bucket'
    ]:
        del hyperparameters[k]

    sagemaker_session = sagemaker.Session()
    inputs = sagemaker.inputs.TrainingInput('s3://bukovec-ml-data/FakeAudio', input_mode='FastFile')

    estimator = Estimator(
        image_uri=args['image_uri'],
        role=sagemaker.get_execution_role(),
        hyperparameters=hyperparameters,
        instance_count=1,
        base_job_name=args['model_name'],
        output_path=args['output_bucket'],
        checkpoint_s3_uri=args['checkpoint_bucket'],
        instance_type=args['instance_type'],
        use_spot_instances=True,
        max_run=24*60*60,
        max_wait=2*24*60*60
    )

    my_tuner = HyperparameterTuner(
        estimator=estimator,  
        objective_metric_name='final_val_score',
        hyperparameter_ranges={
            'bce_weight': ContinuousParameter(1, 8),
            'dropout': ContinuousParameter(0, 0.6),
            'drop_path': ContinuousParameter(0, 0.1),
            'weight_decay': ContinuousParameter(0, 0.1)
        },
        metric_definitions=[
            {'Name': 'final_val_score', 'Regex': 'final_val_score = (\d\.\d+)'},
            {'Name': 'final_val_loss', 'Regex': 'final_val_loss = (\d\.\d+)'},
            {'Name': 'final_val_class_acc', 'Regex': 'final_val_class_acc = (\d\.\d+)'},
            {'Name': 'final_val_seg_prec', 'Regex': 'final_val_seg_prec = (\d\.\d+)'},
            {'Name': 'final_val_sec_acc', 'Regex': 'final_val_seg_acc = (\d\.\d+)'},
            {'Name': 'final_val_sec_f1', 'Regex': 'final_val_seg_f1 = (\d\.\d+)'},
        ],
        max_jobs=20,
        max_parallel_jobs=5,
        random_seed=42
    )
    
    # Start hyperparameter tuning job
    my_tuner.fit(inputs)

    
