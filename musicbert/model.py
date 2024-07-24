import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from transformers import BertModel

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

        weight = split_layer.intermediate.dense.weight
        u, s, v = torch.linalg.svd(weight)

        if rank <= 0 or rank > s.size(0):
            rank = min(64, s.size(0))
        print(f"Adjusted rank: {rank}")

        self.dense_v = nn.Linear(bert.config.hidden_size, rank, bias=False)
        self.dense_v.weight = nn.Parameter(v[:, :rank].contiguous().t())

        self.dense_s = nn.Linear(rank, bert.config.hidden_size, bias=False)
        self.dense_s.weight = nn.Parameter(torch.diag(s[:rank]))

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        print(f"Initial input_ids shape: {input_ids.shape}")
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1) // 88
        print(f"Batch size: {batch_size}, Seq length: {seq_length}")

        # Ensure that input_ids can be reshaped correctly
        if input_ids.size(1) % 88 != 0:
            raise ValueError(f"Input length {input_ids.size(1)} is not divisible by 88")

        input_ids = input_ids.view(batch_size, seq_length, 88)
        transformed_input = self.input_transform(input_ids)
        print(f"Transformed input shape: {transformed_input.shape}")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
            token_type_ids = token_type_ids.view(batch_size, seq_length)
        else:
            token_type_ids = torch.zeros(batch_size, seq_length, device=self.device)
        print(f"Token type IDs shape after view: {token_type_ids.shape}")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            attention_mask = attention_mask.view(batch_size, seq_length)
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        print(f"Attention mask shape after view: {attention_mask.shape if attention_mask is not None else None}")

        hidden_states = self.embeddings(inputs_embeds=transformed_input, token_type_ids=token_type_ids)
        for i, layer in enumerate(self.encoder):
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
            print(f"Hidden states shape after layer {i}: {hidden_states.shape}")

        hidden_states = self.attention(hidden_states)[0]
        hidden_states = self.dense_v(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.dense_s(hidden_states.permute(0, 2, 1))

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
        self.front_ref = rpc.remote("edge", MusicBertFrontDecomposition,
                                    args=(cloud_pretrain_dir, split_num, rank, devices["edge"]))
        self.back_ref = rpc.remote("cloud", MusicBertBackDecomposition,
                                   args=(edge_pretrain_dir, split_num, rank, label_nums, devices["cloud"]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.cpu()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.cpu()
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()

        print(f"Sending input_ids to front_ref: {input_ids.shape}")
        hs = self.front_ref.rpc_sync().forward(input_ids, token_type_ids, attention_mask)
        print(f"Received hs from front_ref: {hs.shape}")
        prediction = self.back_ref.rpc_sync().forward(hs, attention_mask)
        print(f"Received prediction from back_ref: {prediction.shape}")

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
        
        self.input_transform = nn.Linear(88, bert.config.hidden_size)
        print(f"Initialized input_transform with in_features=88, out_features={bert.config.hidden_size}")

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num)])

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        try:
            print(f"Forward method called with input_ids: {input_ids.shape}, token_type_ids: {token_type_ids.shape}, attention_mask: {attention_mask.shape}")
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device).long()  # Ensure token_type_ids is LongTensor
            attention_mask = attention_mask.to(self.device)
            
            batch_size = input_ids.size(0)
            total_length = input_ids.size(1)
            seq_length = total_length // 88
            print(f"Batch size: {batch_size}, Total length: {total_length}, Sequence length: {seq_length}")

            # Verify the total number of elements before and after transformation
            total_elements_before = input_ids.numel()

            input_ids = input_ids.view(batch_size * seq_length, 88)
            print(f"Reshaped input_ids: {input_ids.shape}")

            transformed_input = self.input_transform(input_ids)
            print(f"Transformed input shape: {transformed_input.shape}")

            # Correctly calculate the sequence length after the transformation
            seq_length_transformed = transformed_input.size(0) // batch_size
            transformed_input = transformed_input.view(batch_size, seq_length_transformed, -1)
            print(f"Reshaped transformed input: {transformed_input.shape}")

            # Check the total elements to verify consistency
            total_elements_after = transformed_input.numel()
            print(f"Total elements before: {total_elements_before}, Total elements after: {total_elements_after}")
            assert total_elements_after == total_elements_before * (768 / 88), f"Mismatch in total elements before ({total_elements_before}) and after ({total_elements_after}) transformation."

            # Ensure token_type_ids and attention_mask are reshaped accordingly
            token_type_ids = token_type_ids.view(batch_size, seq_length)
            print(f"Original token_type_ids shape: {token_type_ids.shape}")
            token_type_ids = token_type_ids.view(batch_size, seq_length, 88).view(batch_size * seq_length, 88)
            token_type_ids = token_type_ids.view(batch_size, seq_length_transformed)
            print(f"Reshaped token_type_ids: {token_type_ids.shape}")

            attention_mask = attention_mask.view(batch_size, seq_length)
            print(f"Original attention_mask shape: {attention_mask.shape}")
            attention_mask = attention_mask.view(batch_size, seq_length, 88).view(batch_size * seq_length, 88)
            attention_mask = attention_mask.view(batch_size, seq_length_transformed)
            print(f"Reshaped attention_mask: {attention_mask.shape}")

            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            print(f"Extended attention mask: {extended_attention_mask.shape}")

            print(f"Before embeddings - token_type_ids: {token_type_ids.shape}, transformed_input: {transformed_input.shape}")
            assert transformed_input.size(1) == token_type_ids.size(1), f"Sequence lengths do not match: {transformed_input.size(1)} != {token_type_ids.size(1)}"
            hidden_states = self.embeddings(inputs_embeds=transformed_input, token_type_ids=token_type_ids)
            print(f"After embeddings - hidden_states: {hidden_states.shape}")

            for i, layer in enumerate(self.encoder):
                hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
                print(f"After encoder layer {i} - hidden_states: {hidden_states.shape}")

            return hidden_states.cpu()

        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e

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
        print(f"Back forward received hidden_states: {hidden_states.shape}, attention_mask: {attention_mask.shape}")

        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        print(f"Extended attention mask in back: {extended_attention_mask.shape}")

        for i, layer in enumerate(self.encoder):
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
            print(f"Hidden states after encoder layer {i + split_num}: {hidden_states.shape}")

        hidden_states = self.pooler(hidden_states)
        print(f"Hidden states after pooler: {hidden_states.shape}")

        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)
        print(f"Final prediction shape: {prediction.shape}")

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

class DistMusicBertOri(nn.Module):
    def __init__(self, cloud_pretrain_dir, edge_pretrain_dir, split_num, rank, label_nums, devices):
        super().__init__()
        self.front_ref = rpc.remote("edge", MusicBertFrontOri, args=(cloud_pretrain_dir, split_num, rank, devices["edge"]))
        self.back_ref = rpc.remote("cloud", MusicBertBackOri,
                                   args=(edge_pretrain_dir, split_num, rank, label_nums, devices["cloud"]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        print(f"DistMusicBertOri forward - input_ids: {input_ids.shape}, token_type_ids: {token_type_ids.shape}, attention_mask: {attention_mask.shape}")

        input_ids = input_ids.cpu()
        token_type_ids = token_type_ids.cpu()
        attention_mask = attention_mask.cpu()

        print(f"Sending input_ids to front_ref: {input_ids.shape}")
        hs = self.front_ref.rpc_sync().forward(input_ids, token_type_ids, attention_mask)
        print(f"Received hs from front_ref: {hs.shape}")
        prediction = self.back_ref.rpc_sync().forward(hs, attention_mask)
        print(f"Received prediction from back_ref: {prediction.shape}")

        return prediction

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())

        return remote_params