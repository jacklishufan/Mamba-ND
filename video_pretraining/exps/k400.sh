export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_small_f32_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='<your path here>'
DATA_PATH='<your path here>'

PARTITION='video5'
GPUS=16
GPUS_PER_NODE=8
CPUS_PER_TASK=16

python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
        --nproc_per_node=$1 --master_port=12433 \
        main.py \
        --model mamba3d_small \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ' ' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --output_dir outputs_mint1 \
        --log_dir logs_mint1 \
        --num_frames 16 \
        --num_workers 8 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 50 \
        --lr 4e-4 \
        --drop_path 0.35 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --save_ckpt \
        --dist_eval \
        --test_best \
        --bf16
