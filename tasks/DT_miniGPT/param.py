import argparse
import datetime

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument('--experiment_id', type=str, required=True)
        self.parser.add_argument('--model_name', type=str, default='miniGPT')
        self.parser.add_argument('--features', type=str, default='ResNet-152-imagenet_80_16_mean.tsv')
        self.parser.add_argument('--batch_size', type=int, default=1024)
        self.parser.add_argument('--max_episode_len', type=int, default=30)
        self.parser.add_argument('--learning_rate', type=float, default=6e-4)
        self.parser.add_argument('--betas', type=tuple, default=(0.9, 0.95))
        self.parser.add_argument('--grad_norm_clip', type=float, default=1.0)
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--lr_decay', type=bool, default=True)
        self.parser.add_argument('--warmup_tokens', type=int, default=512*20)
        self.parser.add_argument('--final_tokens', type=int, default=260e9)
        self.parser.add_argument('--ckpt_file', type=str, default=None)
        self.parser.add_argument('--num_workers', type=int, default=12)
        self.parser.add_argument('--feedback_method', type=str, choices=['random', 'teacher', 'random_teacher'],default='random_teacher')
        self.parser.add_argument('--action_level', type=str, default='LLA')
        self.parser.add_argument('--cuda', type=int, choices=range(4),required=True)
        self.parser.add_argument('--reward_strategy', type=int, choices=range(1,8), required=True)
        self.parser.add_argument('--model_type', type=str, default="reward_conditioned")
        self.parser.add_argument('--seed', type=int, default=123)
        self.parser.add_argument('--context_length', type=int, default=5)
        self.parser.add_argument('--epochs', type=int, required=True)
        self.parser.add_argument('--game', type=str, default='Breakout')
        self.parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
        self.parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
        self.parser.add_argument('--train_samples', type=int, default=60000)
        self.parser.add_argument('--save_interval', type=int, default=5)
        self.parser.add_argument('--mode', type=str, choices=['train', 'val', 'debug'], required=True)
        self.parser.add_argument('--fusion_type', type=str, choices=['bert', 'simple', 'attention'], required=True)
        self.parser.add_argument('--target_rtg', type=float, required=True)
        self.parser.add_argument('--dataset_name', type=str, default='right_left_mix_teacher')
        self.parser.add_argument('--notes', type=str, default='')
        self.parser.add_argument('--bert_layers', type=int, default=1) # 1,2, 4, 6
        self.parser.add_argument('--feature_type', type=str, default='fused')

        self.args = self.parser.parse_args()
        #self.warmup_tokens = self.args.warmup_tokens / 512 * self.args.batch_size
        #if self.args.mode == 'val' and self.args.ckpt_file == None:
        #    raise ValueError('Please provide a checkpoint file for validation.')
        self.args.reward_strategy = f"reward_strategy_{self.args.reward_strategy}"
        if self.args.experiment_id == 'time':
            current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            self.args.experiment_id = current_time
        if self.args.mode == 'debug':
            self.args.experiment_id = 'debug'
            self.args.save_interval = 1
            self.args.epochs = 1
            self.args.train_samples = 100

        self.args.final_tokens = 2 * self.args.train_samples * self.args.context_length * 3

param = Param()
args = param.args

# python tasks/DT_miniGPT/train_GPT.py --experiment_id 'bert-layer_1' --cuda 3 --reward_strategy 1 --epochs 15 --fusion_type bert --target_rtg 5 --mode train