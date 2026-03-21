<img src="assets/banner.gif" width="100%"/>

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://dreamdojo-world.github.io/"><strong>Website</strong></a> | 
    <a href="https://arxiv.org/abs/2602.06949"><strong>Paper</strong></a> |
    <a href="https://huggingface.co/nvidia/DreamDojo"><strong>Models</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1"><strong>Datasets</strong></a>
  </p>
</div>

# 💭 DreamDojo

## 🔥 Highlights

*DreamDojo* is an interactive world model that learns from large-scale human videos. In short, we made the following key contributions:

- **A large-scale video dataset.** 44k hours of diverse human egocentric videos, the largest dataset to date for world model pretraining.
- **A foundation world model.** The first robot world model of its kind that demonstrates strong generalization to diverse objects and environments after post-training.
- **A distillation pipeline.** After distillation, our model can achieve long-horizon autoregressive generation, with stable real-time interactions at 10 FPS for over 1 minute.

## 📢 News

- **[2026/02/18]** We released both pretraining and post-training code.
- **[2026/02/18]** We released all pretrained and post-trained checkpoints (2B and 14B).
- **[2026/02/18]** We released the GR-1 post-training datasets and the evaluation sets.
- **[2026/02/09]** We released our [paper](https://arxiv.org/abs/2602.06949) on arXiv.

## 🕹️ Quick Start

- [Setup](https://github.com/NVIDIA/DreamDojo/blob/main/docs/SETUP.md)
- [Latent Action Model Training](https://github.com/NVIDIA/DreamDojo/blob/main/docs/LAM.md)
- [DreamDojo Pretraining](https://github.com/NVIDIA/DreamDojo/blob/main/docs/PRETRAIN.md)
- [DreamDojo Post-Training](https://github.com/NVIDIA/DreamDojo/blob/main/docs/POSTTRAIN.md)
- [DreamDojo Distillation](https://github.com/NVIDIA/DreamDojo/blob/main/docs/DISTILL.md)
- [Evaluation](https://github.com/NVIDIA/DreamDojo/blob/main/docs/EVAL.md)
- [Trouble Shooting](https://github.com/NVIDIA/DreamDojo/blob/main/docs/ISSUES.md)

## ⭐ Citation

If you find our work useful, please consider citing us and giving a star to our repo.

```bibtex
@article{gao2026dreamdojo,
    title={DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos},
    author={Shenyuan Gao and William Liang and Kaiyuan Zheng and Ayaan Malik and Seonghyeon Ye and Sihyun Yu and Wei-Cheng Tseng and Yuzhu Dong and Kaichun Mo and Chen-Hsuan Lin and Qianli Ma and Seungjun Nah and Loic Magne and Jiannan Xiang and Yuqi Xie and Ruijie Zheng and Dantong Niu and You Liang Tan and K.R. Zentner and George Kurian and Suneel Indupuru and Pooya Jannaty and Jinwei Gu and Jun Zhang and Jitendra Malik and Pieter Abbeel and Ming-Yu Liu and Yuke Zhu and Joel Jang and Linxi "Jim" Fan},
    journal={arXiv preprint arXiv:2602.06949},
    year={2026}
}
```

## ⚖️ License

DreamDojo source code is released under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

[![Star History Chart](https://api.star-history.com/svg?repos=NVIDIA/DreamDojo&type=Date)](https://star-history.com/#NVIDIA/DreamDojo&Date)