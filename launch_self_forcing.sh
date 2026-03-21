set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

export WANDB_HTTP_TIMEOUT=300
export WANDB_RETRY_MAX=20
export WANDB_STATS_SAMPLE_RATE_SECONDS=10
export WANDB_STATS_SAMPLES_PER_CORE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# export NCCL_DEBUG=INFO

# CUDA 12 fix: Force PyTorch to use its bundled CUDA libraries
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""  # Clear any preloaded libraries

echo "Running on $NNODES nodes with $NPROC processes per node. This node rank is $NODE_RANK."

export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=8
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-./logs}
export WANDB_API_KEY=${WANDB_API_KEY:?Please set WANDB_API_KEY environment variable}

source .venv/bin/activate

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr $MASTER_ADDR \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
  -- experiment=cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing_no_s3 \
  2>&1 | tee output_train.log