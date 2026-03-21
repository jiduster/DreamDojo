#!/bin/bash
set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1

# CUDA 12 fix: Force PyTorch to use its bundled CUDA libraries
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""  # Clear any preloaded libraries

echo "Running REALTIME inference on $NNODES nodes with $NPROC processes per node."

export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=8
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-./logs}
export WANDB_API_KEY=${WANDB_API_KEY:?Please set WANDB_API_KEY environment variable}

source .venv/bin/activate

# Set CUDA_HOME to help transformer_engine find NVRTC library
export CUDA_HOME=$(python -c "import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0])")

# Add all NVIDIA CUDA libraries to LD_LIBRARY_PATH
NVIDIA_LIB_BASE=$(python -c "import nvidia; print(nvidia.__path__[0])")
TORCH_LIB=$(python -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$NVIDIA_LIB_BASE/cuda_nvrtc/lib:$NVIDIA_LIB_BASE/cublas/lib:$NVIDIA_LIB_BASE/cufft/lib:$NVIDIA_LIB_BASE/cufile/lib:$NVIDIA_LIB_BASE/cusolver/lib:$NVIDIA_LIB_BASE/nccl/lib:$NVIDIA_LIB_BASE/cuda_runtime/lib:$NVIDIA_LIB_BASE/curand/lib:$NVIDIA_LIB_BASE/cudnn/lib:$NVIDIA_LIB_BASE/nvtx/lib:$NVIDIA_LIB_BASE/cuda_cupti/lib:$NVIDIA_LIB_BASE/nvjitlink/lib:$NVIDIA_LIB_BASE/cusparse/lib:$LD_LIBRARY_PATH

python -m cosmos_predict2._src.predict2.interactive.inference.action_video2world_teleop \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
  --experiment=cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing_no_s3 \
  --ckpt_path checkpoints/iter_000006000 \
  --input_frame datasets/eval/video_131_cond_frame.png \
  --fps 10.0 \
  --num_steps 4 \
  --seed 1 \
  --action_source file \
  --action_file datasets/eval/action_131.npy \
  --max_latent_frames 100 \
  --start_frame_idx 0 \
  --warmup \
  --compile_model \
  --no_display \
  --save_output realtime.mp4 \
  "$@"
# Performance options you can add:
#   --warmup              Run warmup iteration (compiles CUDA kernels first)
#   --compile_model       Use torch.compile() for speedup (slow first run)
#   --skip_decode_display N  Only decode every N chunks for display (not used for headless)
#   --fast_mode           Enable ALL optimizations
#
# To use video instead of image frame:
#   Replace --input_frame with:
#   --input_video /path/to/video.mp4 --start_frame_idx 0