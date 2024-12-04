FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

RUN pip3 install sagemaker-training

RUN pip3 install torchaudio torchvision librosa lightning tensorboard pandas s3fs

COPY src/Vim /opt/ml/code/Vim

COPY src/causal-conv1d /opt/ml/code/causal-conv1d

# RUN pip3 install -r /opt/ml/code/Vim/vim/vim_requirements.txt
RUN pip3 install einops timm triton transformers
RUN pip3 install -v /opt/ml/code/causal-conv1d
RUN pip3 install -v /opt/ml/code/Vim/mamba-1p1p1

COPY src/*.py /opt/ml/code/

ENV PYTHONUNBUFFERED=1

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py