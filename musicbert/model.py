# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Your Name
@Date : 2024/6/25
-----------------------------------
"""
import os
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from transformers import BertModel

from config import args

class MusicBertFrontDecomposition(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.input_transform = nn.Linear(88, bert.config.hidden_size)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num - 1)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.attention = split_layer.attention

        # SVD on the weight of the dense layer in the split_layer
        weight = split_layer.intermediate.dense.weight
        u, s, v = torch.linalg.svd(weight)

        # Print SVD results for debugging
        print("SVD u shape:", u.shape)
        print("SVD s shape:", s.shape)
        print("SVD v shape:", v.shape)
        print("Top s values:", s[:5])

        # Validate rank and set to a default value if it's zero
        if rank <= 0 or rank > s.size(0):
            rank = min(64, s.size(0))  # Use a default rank value or maximum allowable
        print(f"Adjusted rank: {rank}")

        # Define the dense layers with proper dimensions
        self.dense_v = nn.Linear(bert.config.hidden_size, rank, bias=False)
        self.dense_v.weight = nn.Parameter(v[:, :rank].contiguous().t())

        self.dense_s = nn.Linear(rank, bert.config.hidden_size, bias=False)
        self.dense_s.weight = nn.Parameter(torch.diag(s[:rank]))

        print("dense_v weight shape:", self.dense_v.weight.shape)
        print("dense_s weight shape:", self.dense_s.weight.shape)

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)

        transformed_input = self.input_transform(input_ids)
        print("Post transform:", transformed_input.shape)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(inputs_embeds=transformed_input)
        print("Post embeddings:", hidden_states.shape)

        for i, layer in enumerate(self.encoder):
            hidden_states = layer(hidden_states)[0]
            print(f"Post layer {i}:", hidden_states.shape)

        hidden_states = self.attention(hidden_states)[0]
        print("Post attention:", hidden_states.shape)

        hidden_states = self.dense_v(hidden_states)
        print("Post dense_v:", hidden_states.shape)

        # Ensuring the shape consistency by explicitly defining the dense_s transformation
        # hidden_states shape: [batch_size, seq_len, rank]
        hidden_states = hidden_states.permute(0, 2, 1)  # Permute for correct matmul
        print("Before dense_s permute:", hidden_states.shape)

        hidden_states = self.dense_s(hidden_states.permute(0, 2, 1))  # Directly multiply with correct permuted dimensions
        print("After dense_s:", hidden_states.shape)

        return hidden_states.cpu()
    
    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class MusicBertBackDecomposition(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = device

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.intermediate_act_fn = split_layer.intermediate.intermediate_act_fn
        self.output = split_layer.output

        weight = split_layer.intermediate.dense.weight
        bias = split_layer.intermediate.dense.bias
        u, s, v = torch.linalg.svd(weight)

        self.dense_u = nn.Linear(in_features=u.size(0), out_features=u.size(0), bias=True)

        self.dense_u.weight = nn.Parameter(u[:, :rank].clone())
        self.dense_u.bias = bias

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.dense_u(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.output.dense(hidden_states)
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = self.output.LayerNorm(hidden_states)

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class DistMusicBertDecomposition(nn.Module):
    def __init__(self, cloud_pretrain_dir, edge_pretrain_dir, split_num, rank, label_nums, devices):
        super().__init__()
        self.front_ref = rpc.remote("cloud", MusicBertFrontDecomposition,
                                    args=(cloud_pretrain_dir, split_num, rank, devices["cloud"]))
        self.back_ref = rpc.remote("edge", MusicBertBackDecomposition,
                                   args=(edge_pretrain_dir, split_num, rank, label_nums, devices["edge"]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hs = self.front_ref.rpc_sync().forward(input_ids, token_type_ids, attention_mask)
        prediction = self.back_ref.rpc_sync().forward(hs, attention_mask)

        return prediction

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())

        return remote_params


class MusicBertFrontOri(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)
        
        # New layer to transform 88-dimensional input to BERT's embedding size
        self.input_transform = nn.Linear(88, bert.config.hidden_size)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num)])

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class MusicBertBackOri(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = device

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class DistMusicBertOri(nn.Module):
    def __init__(self, cloud_pretrain_dir, edge_pretrain_dir, split_num, rank, label_nums, devices):
        super().__init__()
        self.front_ref = rpc.remote("cloud", MusicBertFrontOri, args=(cloud_pretrain_dir, split_num, rank, devices["cloud"]))
        self.back_ref = rpc.remote("edge", MusicBertBackOri,
                                   args=(edge_pretrain_dir, split_num, rank, label_nums, devices["edge"]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hs = self.front_ref.rpc_sync().forward(input_ids, token_type_ids, attention_mask)
        prediction = self.back_ref.rpc_sync().forward(hs, attention_mask)

        return prediction

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs)

        return remote_params