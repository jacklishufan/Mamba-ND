## Data setup

```
--data
----imagenet
--src
--tools

```
## Model Zoo


Checkpoints available at
| Syntax      | Acc         | Weight |
| ----------- | ----------- |--------| 
| Mamba2D-S/8 | 81.7     | [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/in1k/mamba2d_s.pth)
| Mamba2D-B/8 |  83.0   |   [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/in1k/mamba2d_b.pth) |

## Evaluation
```
./tools/dist_test.sh <config> <ckpt_path> <num gpus>
```

## Training

```
./tools/dist_train.sh <config> <num gpus>
```