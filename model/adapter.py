from transformers import BertEncoder, AutoModel, T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import os
import logging
from collections import namedtuple

class PretrainedModel(nn.Module):
    def __init__(self, model_name='t5-small'):
        super(PretrainedModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = self.model.config
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AdapterConfig:
    hidden_act: str = "gelu"
    adapter_initializer_range: float = 0.0002
    chunk_size_feed_forward: int = 0
    attention_probs_dropout_prob: float= 0.1
    hidden_dropout_prob: float=0.1
    initializer_range: float=0.02
    intermediate_size: int=3072
    layer_norm_eps: float=1e-05
    max_position_embeddings: int=514
    num_attention_heads: int=12
    num_hidden_layers: int=2
    num_labels: int=2
    output_attentions: bool=False
    output_hidden_states: bool=False
    torchscript: bool=False
    type_vocab_size: int=1
    vocab_size: int=50265
    def __init__(self, is_decoder, adapter_size):
        self.is_decoder = is_decoder
        self.add_cross_attention = is_decoder
        self.hidden_size = adapter_size

class Adapter(nn.Module):
    def __init__(self, args):
        super(Adapter, self).__init__()
        if type(args) is dict:
            args = namedtuple('AdapterArgs', args.keys())(*args.values())
        self.args = args
        self.down_project = nn.Linear(
            self.args.project_hidden_size,
            self.args.adapter_size,
        ).to(args.device)
        if args.is_decoder:
            self.encoder_down_project = nn.Linear(
                self.args.project_hidden_size,
                self.args.adapter_size,
            ).to(args.device)
        self.encoder = BertEncoder(AdapterConfig(args.is_decoder, args.adapter_size)).to(args.device)
        self.up_project = nn.Linear(self.args.adapter_size, args.project_hidden_size).to(args.device)
        self.layer_norm = nn.LayerNorm(self.args.project_hidden_size).to(args.device)
        self.init_weights()

    def forward(self, hidden_states, encoder_hidden_states=None):
        # Intermediate values of T5 are too large. Normalize them first.
        hidden_states = self.layer_norm(hidden_states)
        # self.down_project.to(hidden_states.device)
        down_projected = self.down_project(hidden_states)

        first_encoder_param = next(self.encoder.parameters())
        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=first_encoder_param.device)
        encoder_attention_mask = torch.ones(input_shape, device=first_encoder_param.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=first_encoder_param.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.args.num_hidden_layers
        if encoder_hidden_states is not None:
            # self.encoder_down_project.to(encoder_hidden_states.device)
            encoder_hidden_states = encoder_hidden_states.to(down_projected.device)
            self.encoder_down_project.to(down_projected.device)
            encoder_hidden_states = self.encoder_down_project(encoder_hidden_states)
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask, encoder_hidden_states=encoder_hidden_states)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class AdapterModel(nn.Module):
    def __init__(self, model, project_hidden_size, adapter_size, model_parallel, need_norm=False):
        super().__init__()
        self.model = model
        self.adapter = nn.Sequential(
            nn.Linear(
                project_hidden_size,
                adapter_size,
            ),
            # nn.Dropout(0.1),
            # nn.GELU(),
            # nn.Linear(
            #     adapter_size,
            #     adapter_size,
            # ),
            # nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(
                adapter_size,
                project_hidden_size,
            ),
        )
        self.need_norm = need_norm
        if need_norm:
            self.adapter_layernorm = nn.LayerNorm(project_hidden_size)
        if model_parallel:
            device = next(self.model.parameters()).device
            self.adapter = self.adapter.to(device)
            if need_norm:
                self.adapter_layernorm = self.adapter_layernorm.to(device)

    def forward(self, *args, **kwargs):
        try:
            output = self.model(*args, **kwargs)
        except RuntimeError as e:
            print(e)
            import pdb; pdb.set_trace()
        x = output[0] if type(output) is tuple else output
        x = x + self.adapter(x)
        if self.need_norm:
            x = self.adapter_layernorm(x)
        return (x, *output[1:]) if type(output) is tuple else x

class AdapterT5Block(nn.Module):
    def __init__(self, block, project_hidden_size, adapter_size, model_parallel, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = block.is_decoder
        block.layer[0].SelfAttention = AdapterModel(
            block.layer[0].SelfAttention, project_hidden_size, adapter_size, model_parallel)
        block.layer[-1].DenseReluDense = AdapterModel(
            block.layer[-1].DenseReluDense, project_hidden_size, adapter_size, model_parallel)
        if self.is_decoder:
            block.layer[1].EncDecAttention = AdapterModel(
                block.layer[1].EncDecAttention, project_hidden_size, adapter_size, model_parallel)
        # block.layer[0] = AdapterModel(block.layer[0], project_hidden_size, adapter_size, model_parallel)
        self.block = block

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)
