export HC3D_SIMULATOR_PATH=$(pwd)

flag="  --model_name miniGPT 
        --features ResNet-152-imagenet_80_16_mean
        --batch_size 64
        --feedback_method teacher
        --action_level LLA
        --rl_reward_strategy reward_strategy_$2
        --model_type reward_conditioned
        --seed 123
        --context_length 30
        --epochs 10
        --game Breakout
        --trajectories_per_buffer 10
        --data_dir_prefix ./dqn_replay/
        --cuda $1"
         

CUDA_VISIBLE_DEVICES="0,1,2,3" python tasks/HC/DT/miniGPT/train_GPT.py $flag
