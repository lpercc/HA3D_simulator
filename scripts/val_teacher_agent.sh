export HA3D_SIMULATOR_PATH=$(pwd)

flag="  --model_name teacher
        --features ResNet-152-imagenet_80_16_mean
        --batch_size 256
        --max_episode_len 20
        --feedback_method teacher
        --action_level LLA
        --cuda $1"
         

CUDA_VISIBLE_DEVICES="0,1,2,3" python tasks/HA/train.py $flag