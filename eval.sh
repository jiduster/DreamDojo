export PYTHONPATH=/data/home/zyh/DreamDojo:$PYTHONPATH
# export HF_HOME=/data/home/zyh/cosmos_cache
source /data/home/zyh/DreamDojo/.venv/bin/activate

python examples/action_conditioned.py \
  -o outputs/action_conditioned/basic \
  --checkpoints-dir /mnt/ceph/ckpt/2B_GR1_post-train \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir /data/home/zyh/dreamdojo_results/gr1_unified_test \
  --num-frames 49 \
  --num-samples 100 \
  --dataset-path datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
  --data-split test \
  --deterministic-uniform-sampling \
  --checkpoint-interval 5000 \
  --infinite