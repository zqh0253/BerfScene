#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training BEV3D on FFHQ-256"
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
GPUSID=$2
DATASET=$3
echo $GPUS, $GPUSID, $DATASET

./scripts/dist_train_jump.sh ${GPUS} ${GPUSID} sgbev3d \
    --job_name='hammer_sgbev3d_256_64_bs4' \
    --seed=0 \
    --resolution=256\
    --neural_rendering_resolution_initial=64 \
    --image_channels=3 \
    --latent_dim=512 \
    --label_dim=0 \
    --pose_dim=25 \
    --gen_pose_cond=true \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --total_img=25_000_000 \
    --batch_size=8 \
    --val_batch_size=4 \
    --eval_at_start=true \
    --eval_interval=5000 \
    --ckpt_interval=1000 \
    --log_interval=2500 \
    --d_lr=0.0018823529411764706 \
    --d_beta_1=0.0 \
    --d_beta_2=0.9905854573074332 \
    --g_lr=0.002 \
    --g_beta_1=0.0 \
    --g_beta_2=0.9919919678228657 \
    --style_mixing_prob=0.0 \
    --r1_interval=16 \
    --r1_gamma=1.0 \
    --pl_weight=0.0 \
    --use_ada=false \
    ${@:4}
