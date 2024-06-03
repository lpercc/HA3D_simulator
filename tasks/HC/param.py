import argparse

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument('--model_name', type=str, default='HCbaseline')
        self.parser.add_argument('--features', type=str, default='ResNet-152-imagenet_80_16_mean')
        self.parser.add_argument('--max_input_length', type=int, default=80)
        self.parser.add_argument('--batch_size', type=int, default=100)
        self.parser.add_argument('--max_episode_len', type=int, default=20)
        self.parser.add_argument('--word_embedding_size', type=int, default=256)
        self.parser.add_argument('--action_embedding_size', type=int, default=32)
        self.parser.add_argument('--hidden_size', type=int, default=512)
        self.parser.add_argument('--bidirection', type=bool, default=False)
        self.parser.add_argument('--dropout_ratio', type=float, default=0.5)
        self.parser.add_argument('--feedback_method', type=str, default='sample')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001)
        self.parser.add_argument('--weight_decay', type=float, default=0.0005)
        self.parser.add_argument('--n_iters', type=int, default=20000)
        self.parser.add_argument('--model_prefix', type=str, default='seq2seq_%s_imagenet')
        self.parser.add_argument('--action_level', type=str, default='LLA')
        self.parser.add_argument('--cuda', type=int, default=3)
        self.parser.add_argument('--teacher_strategy', choices=['global', 'local', 'shortest'],type=str, default='local')
        self.parser.add_argument('--teacher_radius', type=float, default=4.5)
        self.parser.add_argument('--random_rate', type=float, default=0)
        self.args = self.parser.parse_args()
        if self.args.teacher_strategy != 'local':
            self.args.teacher_radius = 0
        self.args.model_prefix = f'seq2seq_train_on_hc_hc3d_sample_{self.args.action_level}_{self.args.teacher_strategy}_{int(self.args.random_rate*100)}_imagenet'
param = Param()
args = param.args
