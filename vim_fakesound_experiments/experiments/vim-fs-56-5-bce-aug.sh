python3 run_experiment.py \
--model_name vim-fs-56-5-bce-aug \
--patch_height 56 \
--patch_width 5 \
--if_augment 1 \
--seq_loss bce \
--input_bucket "s3://bukovec-ml-data/FakeAudio" \
--checkpoint_bucket "s3://bukovec-ml-checkpoints/vim-fs-experiments" \
--image_uri "471112505033.dkr.ecr.us-east-1.amazonaws.com/vim-fakeaudio-experiments" \
--output_bucket "s3://bukovec-ml-data/models" \
--instance_type "ml.g5.48xlarge"
