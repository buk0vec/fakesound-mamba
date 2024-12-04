import sagemaker
from sagemaker.estimator import Estimator

sagemaker_session = sagemaker.Session()
inputs = sagemaker.inputs.TrainingInput('s3://bukovec-ml-data/imagenette2', input_mode='FastFile')

hyperparameters = {
    'learning_rate': 5e-4, 
    'weight_decay': 0.1,
    'warmup_epochs': 50, 
    'warmup_lr': 1e-6, 
    'cosine_epochs': 500, 
    'cosine_t': 500, 
    'cosine_min': 1e-5, 
    'cooldown_epochs': 50, 
    'cooldown_lr': 1e-5,
    'batch_size': 128,
    'num_workers': 7,
    'num_val_workers': 3,
    'persist_workers': 1,
    'persist_val_workers': 0,
    'pin_memory': 1,
    'pin_val_memory': 0,
    'val_batches': 0,
    'ema': 1,
    'model_name': 'vim-small'
}

estimator = Estimator(
    image_uri='471112505033.dkr.ecr.us-east-1.amazonaws.com/vim-training',
    role=sagemaker.get_execution_role(),
    hyperparameters=hyperparameters,
    instance_count=1,
    output_path="s3://bukovec-ml-data/models",
    instance_type='ml.p3.2xlarge',
    use_spot_instances=True,
    max_run=24*60*60,
    max_wait=2*24*60*60
)

estimator.fit(inputs)