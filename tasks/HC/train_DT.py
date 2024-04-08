import json
import os

import torch
from tqdm import tqdm
from transformers import BartModel, BartTokenizer

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

TRAIN_VOCAB = os.path.join(HC3D_SIMULATOR_PATH, "tasks/HC/data/train_vocab.txt")
TRAINVAL_VOCAB = os.path.join(HC3D_SIMULATOR_PATH, "tasks/HC/data/trainval_vocab.txt")
RESULT_DIR = os.path.join(HC3D_SIMULATOR_PATH, "tasks/HC/results/")
SNAPSHOT_DIR = os.path.join(HC3D_SIMULATOR_PATH, "tasks/HC/snapshots/")
PLOT_DIR = os.path.join(HC3D_SIMULATOR_PATH, "tasks/HC/plots/")

IMAGENET_FEATURES = os.path.join(
    HC3D_SIMULATOR_PATH, "img_features/ResNet-152-imagenet_80_16_mean.tsv"
)
MAX_INPUT_LENGTH = 80

features = IMAGENET_FEATURES
batch_size = 100
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = "teacher"  # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
n_iters = 5000 if feedback_method == "teacher" else 20000
model_prefix = "seq2seq_%s_imagenet" % (feedback_method)


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


def train_val():
    """Train on the training set, and validate on seen and unseen splits."""

    setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = HCBatch(
        features, batch_size=batch_size, splits=["train"], tokenizer=tok, device=device
    )

    # Creat validation environments
    val_envs = {
        split: (
            HCBatch(
                features,
                batch_size=batch_size,
                splits=[split],
                tokenizer=tok,
                device=device,
            ),
            Evaluation([split]),
        )
        for split in ["val_seen", "val_unseen"]
    }

    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    encoder = EncoderLSTM(
        len(vocab),
        word_embedding_size,
        enc_hidden_size,
        padding_idx,
        dropout_ratio,
        bidirectional=bidirectional,
    ).cuda()
    decoder = AttnDecoderLSTM(
        Seq2SeqAgent.n_inputs(),
        Seq2SeqAgent.n_outputs(),
        action_embedding_size,
        hidden_size,
        dropout_ratio,
    ).cuda()
    train(train_env, encoder, decoder, n_iters, val_envs=val_envs)


def eval_DT():
    """Init a env to evaluate decision transformer"""
    setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    mconf = GPT1Config(6, 5 * 3, model_type="reward_conditioned", max_timestep=29)
    model = GPT.load(
        os.path.join(
            HC3D_SIMULATOR_PATH,
            "tasks/HC/DT/models/modelsGPT_model_teacher_strategy_6.pth",
        ),
        mconf,
    )

    val_seen_agent = DecisionTransformerAgent(
        val_env, "/home/qid/minghanli/HC3D_simulator/tasks/HC/results", model
    )

    traj = val_seen_agent.rollout()

    # for this trajactory, we need to cut the trajectory when action is

    # Save to json file as a result
    with open(
        os.path.join(
            HC3D_SIMULATOR_PATH, "tasks/HC/results/DT/DT_val_seen_result_2.json"
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
    # train_val()
    # test_submission()
    evaluator = Evaluation(["val_seen"])
    score_summary, _ = evaluator.score(
        os.path.join(
            HC3D_SIMULATOR_PATH, "tasks/HC/results/DT/DT_val_seen_result_2.json"
        )
    )
    print(score_summary)
