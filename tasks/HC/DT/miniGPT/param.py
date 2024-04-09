import argparse

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument('--model_name', type=str, default='miniGPT')
        self.parser.add_argument('--features', type=str, required=True)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--feedback_method', type=str, default='teacher')
        self.parser.add_argument('--action_level', type=str, default='LLA')
        self.parser.add_argument('--cuda', type=int, default=0)
        self.parser.add_argument('--rl_reward_strategy', type=str, default="reward_strategy_1")
        self.parser.add_argument('--model_type', type=str, default="reward_conditioned")
        self.parser.add_argument('--seed', type=int, default=123)
        self.parser.add_argument('--context_length', type=int, default=30)
        self.parser.add_argument('--epochs', type=int, default=5)
        self.parser.add_argument('--game', type=str, default='Breakout')
        self.parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
        self.parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
        
        self.args = self.parser.parse_args()
param = Param()
args = param.args
