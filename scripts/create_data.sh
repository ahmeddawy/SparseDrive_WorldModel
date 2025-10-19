export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0-mini

# python tools/data_converter/nuscenes_converter.py nuscenes \
#     --root-path /home/oem/Practice/PHD/nuscenes_full/v1.0-trainval \
#     --canbus  /home/oem/Practice/PHD/nuscenes_full/v1.0-trainval \
#     --out-dir  /home/oem/Practice/PHD/nuscenes_full/infos/ \
#     --extra-tag nuscenes \
#     --version v1.0

