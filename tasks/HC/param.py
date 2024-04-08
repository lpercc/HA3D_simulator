import argparse

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument('--model_name', type=str, default='R2Rbaseline')
        self.parser.add_argument('--features', type=str, required=True)
        self.parser.add_argument('--max_input_length', type=int, default=80)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--max_episode_len', type=int, default=20)
        self.parser.add_argument('--word_embedding_size', type=int, default=256)
        self.parser.add_argument('--action_embedding_size', type=int, default=32)
        self.parser.add_argument('--hidden_size', type=int, default=512)
        self.parser.add_argument('--bidirectional', type=bool, default=False)
        self.parser.add_argument('--dropout_ratio', type=float, default=0.5)
        self.parser.add_argument('--feedback_method', type=str, default='sample')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001)
        self.parser.add_argument('--weight_decay', type=float, default=0.0005)
        self.parser.add_argument('--n_iters', type=int, default=5000)
        self.parser.add_argument('--model_prefix', type=str, default='seq2seq_%s_imagenet')
        self.parser.add_argument('--action_level', type=str, default='sLLA')
        self.parser.add_argument('--cuda', type=int, default=0)
        self.parser.add_argument('--rl_reward_strategy', type=str, default="teacher_strategy_1")
        self.parser.add_argument('--model_type', type=str, default="reward_conditioned")
param = Param()
args = param.args
