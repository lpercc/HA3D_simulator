#!/bin/bash

export HA3D_SIMULATOR_PATH=$(pwd)

flag="--model_name miniGPT 
        --features ResNet-152-imagenet_80_16_mean
        --batch_size 1024
        --feedback_method random_teacher
        --action_level LLA
        --reward_strategy reward_strategy_$1
        --context_length 5
        --max_episode_len 30
        --ckpt_file model_epoch_10.pth
        --epochs $5
        --cuda $2
        --mode $3
        --target_rtg $4
        --save_interval 5
        --fusion_type $6"
         

CUDA_VISIBLE_DEVICES="0,1,2,3" python tasks/DT_miniGPT/train_GPT.py $flag
