# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Your Name
@Date : 2024/6/25
-----------------------------------
"""
import os
import argparse
import logging
import sys

# system settings
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
root = os.path.dirname(__file__)
sys.path.append(root)

parser = argparse.ArgumentParser()
# path settings
parser.add_argument("--edge_pretrain_dir", type=str, default="/home/brad/models/splitfinetuning/pretrain/bert-base-uncased")
parser.add_argument("--cloud_pretrain_dir", type=str, default="/home/brad/models/splitfinetuning/pretrain/bert-base-uncased")
parser.add_argument("--data_dir", type=str, default="/home/brad/models/splitfinetuning/data/clean_midi")
parser.add_argument("--save_path", type=str, default="music_model.pt")
parser.add_argument("--log_path", type=str, default="music_model.log")

# hyperparameter settings
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--dev_batch_size", type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--warm_up_pct", type=float, default=0.1)
parser.add_argument("--cuda", type=int, default=0, help="the index of gpus")
parser.add_argument("--split", type=int, default=12, help="indicate which layer of the bert to be split")
parser.add_argument("--rank", type=int, default=8, help="the rank of svd decomposition of the split layer")

args = parser.parse_args()

args.train_path = os.path.join(args.data_dir, "train.npy")
args.dev_path = os.path.join(args.data_dir, "dev.npy")
args.test_path = os.path.join(args.data_dir, "test.npy")

# log settings
with open(args.log_path, 'a', encoding="utf-8") as f:
    f.write('\n')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(args.log_path, 'a', encoding='utf-8')
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)
