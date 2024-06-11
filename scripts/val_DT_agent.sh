export HA3D_SIMULATOR_PATH=$(pwd)

flag="  --model_name miniGPT 
        --features ResNet-152-imagenet_80_16_mean
        --batch_size 100
        --feedback_method teacher
        --action_level LLA
        --rl_reward_strategy reward_strategy_$2
        --context_length 30
        --cuda $1"

CUDA_VISIBLE_DEVICES="0,1,2,3" python tasks/HA/val_DT.py $flag