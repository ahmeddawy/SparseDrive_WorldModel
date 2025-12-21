# #!/usr/bin/env bash

# # Test World Model Stage 2
# # Usage: bash scripts/test_wm.sh

# bash ./tools/dist_test.sh \
#     projects/configs/sparsedrive_small_stage2.py \
#     /home/oem/Practice/sparsedrive_law/SparseDrive_LAW/work_dirs/sparsedrive_stage2_wm_optimized/iter_93760.pth \
#     1 \
#     --deterministic \
#     --eval bbox
#     # --result_file ./work_dirs/sparsedrive_small_stage2_wm_optimized_v2/results.pkl



bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2_wm_optimized_v2.py \
    /home/oem/Practice/sparsedrive_law/SparseDrive_LAW/work_dirs/sparsedrive_stage2_wm_optimized/iter_93760.pth \
    1 \
    --deterministic \
    --eval bbox
