# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Your Name
@Date : 2024/6/25
-----------------------------------
"""
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc

from config import args, logger
from model import DistBertDecomposition, DistBertOri, DistBertFFN
from data import MidiDataset

devices = {
    "edge": "cpu",
    "cloud": "cuda"
}

rpc.init_rpc("edge", rank=0, world_size=2)

train_data = MidiDataset(args.data_dir)
train_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=train_data.collate, shuffle=True)

dev_data = MidiDataset(args.data_dir)
dev_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, collate_fn=dev_data.collate)

classifier = DistBertDecomposition(args.cloud_pretrain_dir,
                                   args.edge_pretrain_dir,
                                   args.split,
                                   args.rank,
                                   train_data.label_nums,
                                   devices)
# classifier = DistBertFFN(args.cloud_pretrain_dir,
#                          args.edge_pretrain_dir,
#                          args.split,
#                          args.rank,
#                          train_data.label_nums,
#                          devices)

criterion = nn.CrossEntropyLoss()

opt = DistributedOptimizer(AdamW, classifier.parameter_rrefs(), lr=args.learning_rate)

for e in range(args.epoch):
    classifier.train()
    logger.info("epoch starts")
    for i, batch in tqdm(enumerate(train_loader), desc="training", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(devices["cloud"])
        batch_token_type_ids = batch["batch_token_type_ids"].to(devices["cloud"])
        batch_attention_mask = batch["batch_attention_mask"].to(devices["cloud"])
        y = batch["batch_labels"].to(devices["edge"])

        with dist_autograd.context() as context_id:
            y_hat = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                               attention_mask=batch_attention_mask).to(devices["edge"])
            loss = criterion(y_hat, y)
            dist_autograd.backward(context_id, [loss])
            opt.step(context_id)

    logger.info("epoch ends")
    classifier.eval()

    dev_loss = 0.0
    right_num = 0
    for batch in tqdm(dev_loader, desc="eval dev data", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(devices["cloud"])
        batch_token_type_ids = batch["batch_token_type_ids"].to(devices["cloud"])
        batch_attention_mask = batch["batch_attention_mask"].to(devices["cloud"])
        y = batch["batch_labels"].to(devices["edge"])

        with torch.no_grad():
            y_hat = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                               attention_mask=batch_attention_mask).to(devices["edge"])
            loss = criterion(y_hat, y)
            pred = torch.argmax(y_hat, dim=-1)
            loss = loss.detach().cpu().item()
            right = torch.eq(y, pred).sum().detach().item()
        dev_loss += loss * len(batch)
        right_num += right

    all_num = len(dev_data)
    logger.info(f"epoch: {e} dev set loss: {dev_loss / all_num} acc: {right_num / all_num}")

    torch.save(classifier.state_dict(), os.path.join(args.save_dir, f'classifier_epoch_{e}.pt'))

rpc.shutdown()
