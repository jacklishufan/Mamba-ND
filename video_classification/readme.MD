## Data setup

```
--data
----hmdb51
----ucf101
--src
--tools

```
## Model Zoo

| Syntax      | Acc | Weight |
| ----------- | ----------- | ----------- |
| UCF-101    | 89.6       |[weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/video/ucf101/ucf101.pth)
| HMDB-51 |  60.9  | [weight](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/video/hmdb51/hmdb51.pth)


## Evaluation
```
./tools/dist_test.sh <config> <ckpt_path> <num gpus>
```

## Training

1. Download ImageNet Pretraiend Weights of Mamba-2D-S/16 [here](https://huggingface.co/jacklishufan/Mamba-ND/blob/main/video/mamba_2d_s_16_pretrained.pth)

2. replace the path of `pretrained='mamba2d_s_16.pth'` in config to the correct path

3.

```
./tools/dist_train.sh <config> <num gpus>
```