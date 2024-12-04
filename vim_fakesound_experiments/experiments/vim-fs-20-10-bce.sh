python3 run_experiment.py \
--model_name vim-fs-20-10-bce \
--seq_loss bce \
--input_bucket "s3://bukovec-ml-data/FakeAudio" \
--checkpoint_bucket "s3://bukovec-ml-checkpoints/vim-fs-experiments" \
--image_uri "471112505033.dkr.ecr.us-east-1.amazonaws.com/vim-fakeaudio-experiments" \
--output_bucket "s3://bukovec-ml-data/models" \
--instance_type "ml.g5.48xlarge"
