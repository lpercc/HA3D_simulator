import argparse

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument('--model_name', type=str, default='miniGPT')
        self.parser.add_argument('--features', type=str, default='img_features/ResNet-152-imagenet_80_16_mean.tsv')
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--max_episode_len', type=int, default=30)
        self.parser.add_argument('--learning_rate', type=int, default=6e-4)
        self.parser.add_argument('--betas', type=tuple, default=(0.9, 0.95))
        self.parser.add_argument('--grad_norm_clip', type=float, default=1.0)
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--lr_decay', type=bool, default=True)
        self.parser.add_argument('--warmup_tokens', type=int, default=512*20)
        self.parser.add_argument('--final_tokens', type=int, default=260e9)
        self.parser.add_argument('--ckpt_file', type=str, default=None)
        self.parser.add_argument('--num_workers', type=int, default=12)
        self.parser.add_argument('--feedback_method', type=str, default='teacher')
        self.parser.add_argument('--action_level', type=str, default='LLA')
        self.parser.add_argument('--cuda', type=int, default=2)
        self.parser.add_argument('--reward_strategy', type=str, default="reward_strategy_3")
        self.parser.add_argument('--model_type', type=str, default="reward_conditioned")
        self.parser.add_argument('--seed', type=int, default=123)
        self.parser.add_argument('--context_length', type=int, default=30)
        self.parser.add_argument('--epochs', type=int, default=5)
        self.parser.add_argument('--game', type=str, default='Breakout')
        self.parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
        self.parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
        self.parser.add_argument('--train_samples', type=int, default=60000)
        self.parser.add_argument('--save_interval', type=int, default=5)
        self.parser.add_argument('--validation', type=bool, default=False)
        self.parser.add_argument('--fusion_type', type=str, default='simple')

        
        self.args = self.parser.parse_args()
        self.args.final_tokens = 2 * self.args.train_samples * self.args.context_length * 3
param = Param()
args = param.args
