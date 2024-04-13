export HC3D_SIMULATOR_PATH=$(pwd)

flag="--model_name miniGPT 
        --features ResNet-152-imagenet_80_16_mean
        --batch_size 1024
        --feedback_method random
        --action_level LLA
        --reward_strategy reward_strategy_$1
        --context_length 5
        --max_episode_len 30
        --ckpt_file model_epoch_10.pth
        --save_interval 5
        --epochs 15
        --cuda $2
        --mode $3"
         

CUDA_VISIBLE_DEVICES="0,1,2,3" python tasks/DT_miniGPT/train_GPT.py $flag
