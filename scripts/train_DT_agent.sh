export HC3D_SIMULATOR_PATH=$(pwd)

flag="--model_name miniGPT 
        --features ResNet-152-imagenet_80_16_mean
        --batch_size 256
        --feedback_method teacher_random
        --action_level LLA
        --reward_strategy reward_strategy_$2
        --context_length 5
        --max_episode_len 30
        --epochs 50
        --cuda $1"
         

CUDA_VISIBLE_DEVICES="0,1,2,3" python tasks/DT_miniGPT/train_GPT.py $flag
