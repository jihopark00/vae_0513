#!/bin/bash
export NCCL_P2P_DISABLE="1"
ENTITY="qkrwlgh0314"
PROJECT="levae"
# WANDB_KEY="4ab8d4a0db9aec6c8095"
WANDB_KEY="4ab8d4a0db9aec6c80956ccf58616de15392a463"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

EXP_NAME=0512_ae_base_lpips
EXPS_DIR=exps
# CONFIG=configs/default.yaml
CONFIG=$EXPS_DIR/$EXP_NAME/config.yaml
DATA_PATH=/dataset/imagenet/train
NPROC=4

torchrun --standalone --nproc_per_node=$NPROC train_reconstruction.py \
    --config $CONFIG \
    --output_dir $EXPS_DIR/$EXP_NAME \
    --print_freq 50 \
    --eval_freq 50 \
    --vis_freq 5 \
    --save_freq 1 \
    --auto_resume \
    --keep_n_ckpts 1 \
    --milestone_interval 50 \
    --online_eval \
    --num_images 50000 \
    --fid_stats_path data/fid_stats/val_fid_statistics_file.npz \
    --eval_bsz 256 \
    --data_path $DATA_PATH \
    --num_classes 1000 \
    --num_workers 10 \
    --project $PROJECT \
    --entity $ENTITY \
    --exp_name $EXP_NAME \
    --enable_wandb \
    --wandb_key $WANDB_KEY
