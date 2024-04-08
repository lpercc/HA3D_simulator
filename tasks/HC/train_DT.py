import json
import os

import torch
from tqdm import tqdm
from transformers import BartModel, BartTokenizer
from param import args
from utils import (
    Tokenizer,
    build_vocab,
    check_agent_status,
    padding_idx,
    read_vocab,
    timeSince,
    write_vocab,
)

HC3D_SIMULATOR_PATH = os.environ.get("HC3D_SIMULATOR_PATH")

TRAIN_VOCAB = os.path.join(HC3D_SIMULATOR_PATH, f'tasks/HC/data/{args.model_name}/train_vocab.txt')
TRAINVAL_VOCAB = os.path.join(HC3D_SIMULATOR_PATH, f'tasks/HC/data/{args.model_name}/trainval_vocab.txt')
RESULT_DIR = os.path.join(HC3D_SIMULATOR_PATH, f'tasks/HC/results/{args.model_name}/')
SNAPSHOT_DIR = os.path.join(HC3D_SIMULATOR_PATH, f'tasks/HC/snapshots/{args.model_name}/')
PLOT_DIR = os.path.join(HC3D_SIMULATOR_PATH, f'tasks/HC/plots/{args.model_name}/')

MAX_INPUT_LENGTH = args.max_input_length

features = os.path.join(HC3D_SIMULATOR_PATH, f'img_features/{args.features}.tsv')
batch_size = args.batch_size
max_episode_len = args.max_episode_len
word_embedding_size = args.word_embedding_size
action_embedding_size = args.action_embedding_size
hidden_size = args.hidden_size
bidirectional = args.bidirection
dropout_ratio = args.dropout_ratio
feedback_method = args.feedback_method # teacher or sample
learning_rate = args.learning_rate
weight_decay = args.weight_decay
n_iters = args.n_iters
model_prefix = args.model_prefix % (feedback_method)

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=["train"]), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(
            build_vocab(splits=["train", "val_seen", "val_unseen"]), TRAINVAL_VOCAB
        )

def eval_DT():
    """Init a env to evaluate decision transformer"""
    setup()
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    tok = BartTokenizer.from_pretrained("facebook/bart-base")
    embedding_model = BartModel.from_pretrained("facebook/bart-base")
    # train_env = HCBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok, text_embedding_model=embedding_model, device=device)

    # Create Validation Environments
    val_env = HCBatch(
        features,
        batch_size=batch_size,
        splits=["val_seen"],
        tokenizer=tok,
        text_embedding_model=embedding_model,
        device=device,
    )

    # load models
    mconf = GPT1Config(6, 5 * 3, model_type=args.model_type, max_timestep=max_episode_len)
    model = GPT.load(
        os.path.join(
            HC3D_SIMULATOR_PATH,
            f"tasks/HC/DT/models/modelsGPT_model_{args.rl_reward_strategy}.pth",
        ),
        mconf,
    )

    val_seen_agent = DecisionTransformerAgent(
        val_env, RESULT_DIR, model
    )

    traj = val_seen_agent.rollout()

    # for this trajactory, we need to cut the trajectory when action is

    # Save to json file as a result
    with open(
        os.path.join(
           RESULT_DIR, f"DT_val_seen_result_{args.rl_reward_strategy}.json"
        ),
        "w",
    ) as f:
        json.dump(traj, f)


if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(RESULT_DIR)):
        os.makedirs(os.path.dirname(RESULT_DIR))
    if not os.path.exists(os.path.dirname(SNAPSHOT_DIR)):
        os.makedirs(os.path.dirname(SNAPSHOT_DIR))
    if not os.path.exists(os.path.dirname(PLOT_DIR)):
        os.makedirs(os.path.dirname(PLOT_DIR))
    eval_DT()
    evaluator = Evaluation(["val_seen"])
    score_summary, _ = evaluator.score(
        os.path.join(
            f"DT_val_seen_result_{args.rl_reward_strategy}.json"
        )
    )
    print(score_summary)
