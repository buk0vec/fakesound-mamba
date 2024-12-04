python3 run_experiment.py \
--model_name vim-fs-20-10-bce-sm4-resid-reg \
--seq_loss bce \
--drop_path 0.01 \
--weight_decay 0.05 \
--dropout 0.2 \
--if_seq_mamba 1 \
--if_seq_residual 1 \
--seq_mamba_depth 4 \
--input_bucket "s3://bukovec-ml-data/FakeAudio" \
--checkpoint_bucket "s3://bukovec-ml-checkpoints/vim-fs-experiments" \
--image_uri "471112505033.dkr.ecr.us-east-1.amazonaws.com/vim-fakeaudio-experiments" \
--output_bucket "s3://bukovec-ml-data/models" \
--instance_type "ml.g5.48xlarge"
