#!/bin/bash
set -x

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training VolumeGAN on FFHQ-256."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/ffhq256.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2
NeRFRES=32
RES=256
./scripts/dist_train.sh ${GPUS} volumegan \
    --job_name='volumegan_ffhq_256' \
    --seed=0 \
    --resolution=${RES} \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --val_max_samples=-1 \
    --total_img=25_000_000 \
    --batch_size=8 \
    --val_batch_size=16 \
    --train_data_mirror=true \
    --data_workers=3 \
    --data_pin_memory=true \
    --data_prefetch_factor=2 \
    --data_repeat=500 \
    --train_data_mirror=true \
    --g_init_res=${NeRFRES} \
    --latent_dim=512 \
    --d_fmaps_factor=1.0 \
    --g_fmaps_factor=1.0 \
    --d_mbstd_groups=4 \
    --g_num_mappings=8 \
    --d_lr=0.002 \
    --g_lr=0.002 \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --style_mixing_prob=0.9 \
    --r1_interval=16 \
    --r1_gamma=10.0 \
    --pl_interval=4 \
    --pl_weight=2.0 \
    --pl_decay=0.01 \
    --pl_batch_shrink=2 \
    --g_ema_img=20000 \
    --eval_at_start=false \
    --eval_interval=6400 \
    --ckpt_interval=6400 \
    --log_interval=128 \
    --enable_amp=false \
    --use_ada=false \
    --num_fp16_res=128\
    --logger_type=normal \
    --use_pg=true \
    ${@:3}
