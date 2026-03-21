# Troubleshooting

We will continue to improve this repository. If you encounter any issues, please feel free to let us know!

Make sure you have exported the following variables in the training script. Missing these may cause unstable training performance.

```
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1
```

---

<= Previous: [[Evaluation](https://github.com/NVIDIA/DreamDojo/blob/main/docs/EVAL.md)]