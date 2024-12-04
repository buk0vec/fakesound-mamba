# fakesound-mamba

This is the repository for the CS230 Project "Exploring General Deepfake Audio Detection with Vision Mamba". Here are some of the things you will find in this repository:

`vim_fakesound_experiments/`: Experiments applying Vision Mamba to FakeSound dataset
  - `MobileNet-Baseline.ipynb`: MobileNet baseline code
  - `*.ipynb`: Number crunching
  - `fcn_resnet_eval_local.py`: Script use to evaluate baseline model on test set
  - `Dockerfile`: For building AWS Sagemaker training container
  - `stats`: csvs containing raw predictions and ground truths for models evaluated on test set
  - `tuning/`: scripts for running hyperparameter optimization on AWS Sagemaker
  - `experiments/`: scripts for running training experiments on AWS Sagemaker
  - `src/`: core code
    - `datasets.py`: Custom dataset classes for both local and S3-hosted FakeSound datasets
    - `ema.py`: Code modified from [here](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py) for EMA. We did not get to test this approach with our models.
    - `models.py`: Pytorch lightning wrapper around ViM
    - `save_eval_local.py`: Used for evaluating models against test set, saves ground truths and predictions to files.
    - `schedulers.py`: Implementation of a Cosine LR scheduler w/ warmup and cooldown periods in pure Pytorch. 
    - `train.py`: Main training script
    - `transforms.py`: spectrogram creation and augmentation
    - `utils.py`: Utils for converting ViM checkpoints to trainable models
    - `causal-conv1d/`: An earlier version of the [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) library, which is required by ViM but they package the wrong version with their repository.
    - `Vim/`: The modified [Vision Mamba](https://github.com/kyegomez/VisionMamba) repository
      - `vim/models_mamba.py`: The main file with the ViM model. We modify this file to support sequential outputs along with the classification task. 

`vim_imagenette_training/`: Inital experiments in pretraining Vision Mamba from scratch based on a subsection of ImageNet (pursued in milestone but not final project)

`Downloader.py`: AudioCaps downloader as taken from [here](https://github.com/MorenoLaQuatra/audiocaps-download/blob/main/audiocaps_download/Downloader.py)

## Requirements

I think all the requirements are in `requirements.txt`, but make sure to editable install `vim_fakesound_experiments/src/causal-conv1d` and `vim_fakesound_experiments/src/Vim/mamba-1p1p1` if you want to run this code yourself. I was running this is the most awful conda environment you've ever seen, so I'd rather not use those requirements.