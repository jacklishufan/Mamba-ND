# Ofiicial Implementation for Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data

[Paper](https://arxiv.org/abs/2402.05892)

## Model Zoo

### Image Classification 
Checkpoints available at
| Syntax      | Acc         | Weight |
| ----------- | ----------- |--------| 
| Mamba2D-S/8 | 81.7     | [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/in1k/mamba2d_s.pth)
| Mamba2D-B/8 |  83.0   |   [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/in1k/mamba2d_b.pth) |

### Video Classification 

| Syntax      | Acc | Weight |
| ----------- | ----------- | ----------- |
| UCF-101    | 89.6       |[weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/video/ucf101/ucf101.pth)
| HMDB-51 |  60.9  | [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/video/hmdb51/hmdb51.pth)

### 3D Segmentation
| Syntax      | Feature Size |  Dice         | Weight |
| ----------- | ----------- |  ----------- |--------| 
| Mamba3D-S/16 |32|  83.1  |   [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/btcv/mamband-s.pt) |
| Mamba3D-S+/16 |32|  83.9  |   [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/btcv/mamband-s_plus.pt) |
| Mamba3D-B/16 |32|  82.7  |   [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/btcv/mamband-b-32.pt) |
| Mamba3D-B/16 |64|  84.7  |   [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/btcv/mamband-b-64.pt) |

## Environment Setup


```
pip install causal-conv1d>=1.2.0
git install git+https://github.com/state-spaces/mamba.git
```

For image classification, [mmpretrain](https://mmpretrain.readthedocs.io/en/latest/) is required. For video classification, [mmaction](https://mmpretrain.readthedocs.io/en/latest/) is required.  Please see offical documentation for installation instructions.


## Training

Please see refer to the following instructions for each task:

[Image classification](image_classification/readme.MD)
[Video classification](video_classification/readme.MD)
[3D segmentation](btcv/readme.MD)

## Citation
```
@article{li2024mamba,
  title={Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data},
  author={Li, Shufan and Singh, Harkanwar and Grover, Aditya},
  journal={arXiv preprint arXiv:2402.05892},
  year={2024}
}
```