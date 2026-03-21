# Distillation

The distillation pipeline converts a post-trained DreamDojo teacher model into a fast, causal student model capable of long-horizon autoregressive generation at 10 FPS. The pipeline consists of three stages:

1. **Teacher Generation** — Generate multi-step denoising targets from the teacher model.
2. **Warmup** — Train the causal student architecture to match the teacher's outputs.
3. **Self-Forcing** — Finetune the student with its own autoregressive predictions to reduce error accumulation.

After distillation, you can run offline inference to generate videos from a dataset of action sequences, or real-time inference for interactive teleoperation.

## Teacher Generation

Generate denoising targets from the teacher model at few-step noise levels. This pre-computes the supervision for warmup training.

```bash
bash launch_teacher_gen.sh
```

## Warmup Training

Train the causal student network to match the teacher's denoising outputs. This initializes the student before self-forcing.

```bash
bash launch_warmup.sh
```

The warmup experiment configs are defined in `cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_warmup.py`.

## Self-Forcing Distillation

Finetune the student model with its own autoregressive rollouts to improve long-horizon stability, using the teacher model to provide the score during DMD distillation.

```bash
bash launch_self_forcing.sh
```

The self-forcing experiment configs are defined in `cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_self_forcing.py`.

## Inference

Generate videos conditioned on pre-recorded action sequences via:

```bash
bash launch_student_inference.sh
```

Key arguments:
- `--experiment`: Self-forcing experiment config name, which should match the one used during distillation.
- `--ckpt_path`: Path to the distilled checkpoint.
- `--input_json`: Path to a JSON file containing evaluation entries (each entry specifies a video path, actions, and metadata).

### Real-Time Teleoperation

Run the distilled model interactively with live action inputs (e.g., teleoperation):

```bash
bash launch_student_inference_teleop.sh
```

Key arguments:
- `--ckpt_path`: Path to the distilled checkpoint.
- `--input_frame`: Path to the initial conditioning frame (PNG image).
- `--action_source`: Action input source (`file` for pre-recorded, or other sources for live input).
- `--action_file`: Path to a `.npy` file containing actions (when using `file` source).
- `--max_latent_frames`: Maximum number of latent frames to generate.
- `--fps`: Target generation framerate (default: 10.0).
- `--save_output`: Path to save the generated video (e.g., `output.mp4`).

---

<= Previous: [[DreamDojo Post-Training](https://github.com/NVIDIA/DreamDojo/blob/main/docs/POSTTRAIN.md)]

=> Next: [[Evaluation](https://github.com/NVIDIA/DreamDojo/blob/main/docs/EVAL.md)]
