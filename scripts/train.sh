## stage1 (skipped as we have the checkpoint)
# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_stage1.py \
#    1 \
#    --deterministic

## stage2 (using pre-trained stage1 checkpoint)
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage2.py \
   1 \
   --deterministic