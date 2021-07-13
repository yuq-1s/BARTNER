import torch
from transformers import T5Model, T5Tokenizer
# from transformers import T5Stack, T5Model, T5Tokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import logging

EOS_ID = 1
PAD_ID = 0

class PromptFT5Encoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        # assert isinstance(encoder, T5Stack)
        self.t5_encoder = encoder
        original_embed = encoder.embed_tokens.weight
        self.soft_prompt_embed = nn.Embedding(1, original_embed.size(1))
        self.soft_prompt_embed.weight.data = encoder.embed_tokens.weight[32127].clone().detach().to(original_embed.device)
        self.soft_prompt_embed.requires_grad_(True)
        # self.register_buffer('soft_prompt_embed', soft_prompt_embed)

    def forward(self, src_tokens, src_seq_len):
        embeddings = self.t5_encoder.embed_tokens(src_tokens)
        prompt_embeddings = self.soft_prompt_embed.weight.repeat((src_tokens.size(0), 1, 1))
        embeddings = torch.cat([prompt_embeddings, embeddings], dim=1)
        mask = seq_len_to_mask(src_seq_len+1, max_len=embeddings.size(1))
        dict = self.t5_encoder(inputs_embeds=embeddings, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states

class FT5Encoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        # assert isinstance(encoder, T5Stack)
        self.t5_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.t5_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states

class FT5Decoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp):
        super().__init__()
        # assert isinstance(decoder, T5Stack)
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        mapping = torch.LongTensor([EOS_ID]+label_ids) # T5 has no bos; eos is 1
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        if use_encoder_mlp:
            hidden_size = decoder.embed_tokens.weight.size(1)
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

        logging.warning("FIXME: This version of T5Decoder forces `first` to be None.")

    def forward(self, tokens, state):
        encoder_outputs = state.encoder_output

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(PAD_ID).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        self.mapping = self.mapping.to(mapped_tokens.device)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        first = None
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = self._shift_right(tokens)
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                return_dict=True)
        else:
            if state.past_key_values is None:
                tokens = torch.full(tokens.shape, PAD_ID, device=tokens.device)
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        # eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[EOS_ID:EOS_ID+1])  # bsz x max_len x 1

        hidden_state = hidden_state.to(self.decoder.embed_tokens.weight.device)
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_end_id:self.label_end_id+1])  # bsz x max_len x 1

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output
        src_outputs = src_outputs.to(self.decoder.embed_tokens.weight.device)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        if hasattr(self, 'encoder_mlp'):
            self.encoder_mlp.to(src_outputs.device)
            src_outputs = self.encoder_mlp(src_outputs)

        # If prompt is added, remove it for decoder
        if src_tokens.shape != mask.shape:
            mask = mask[:, 1:]
            src_outputs = src_outputs[:, 1:]
        mask = mask.unsqueeze(1).__or__(src_tokens.eq(EOS_ID).cumsum(dim=1).bool().unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, :1] = eos_scores
        logits[:, :, 1:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return self(tokens, state)[:, -1]

    def _shift_right(self, input_ids):
        assert self.pad_token_id is not None
        decoder_start_token_id = self.pad_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.pad_token_id)
        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
        return shifted_input_ids

def fix_loaded_state_dict(trained_state_dict):
    for k, v in trained_state_dict.items():
        if k == 'seq2seq_model.decoder.mapping':
            yield 'shared.weight', trained_state_dict['seq2seq_model.encoder.t5_encoder.embed_tokens.weight']
        elif 'encoder' in k:
            i = k.rfind('encoder')
            yield k[i:], v
        elif 'decoder' in k:
            i = k.rfind('decoder')
            yield k[i:], v

class T5Seq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False, use_prompt=False, checkpoint_path=None, model_parallel=False):
        model = T5Model.from_pretrained(bart_model, mirror='tuna')
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        # FIXME: Speed up T5: T5's vocab of 32128 has no need to resize_token_embeddings here
        # model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        if checkpoint_path is not None:
            trained = torch.load(checkpoint_path)
            if hasattr(trained, 'state_dict'):
                trained = trained.state_dict()
            if trained.keys() != model.state_dict().keys():
                trained = dict(fix_loaded_state_dict(trained))
            model.load_state_dict(trained)
            logging.info(f"Loading {checkpoint_path} succeeded.")
        if model_parallel:
            model.parallelize()
        encoder = model.encoder
        decoder = model.decoder

        __normalize = lambda x: (x - x.mean()) / x.std() * 0.4356 - 0.0094

        _tokenizer = T5Tokenizer.from_pretrained(bart_model, mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models', local_files_only=True)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                if not index>=num_tokens:
                    logging.warning(f"special token {token} has index {index}, which is larger than {num_tokens}, which is possible for T5 though. See https://github.com/huggingface/transformers/issues/4875")
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = __normalize(embed)

        if use_prompt:
            encoder = PromptFT5Encoder(encoder)
        else:
            encoder = FT5Encoder(encoder)
        if decoder_type is None:
            decoder = FT5Decoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids, use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")

        decoder.decoder.embed_tokens.weight.data[decoder.label_end_id] = __normalize(decoder.decoder.embed_tokens.weight.data[EOS_ID])

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, first, tgt_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new

# class T5Seq2SeqModel(OldT5Seq2SeqModel):
#     def __init__(self, encoder, decoder):
#         super().__init__(encoder, decoder)
#         special_token = 32127
#         special_token = encoder.t5_encoder.embed_tokens.weight[-1].clone().detach()
#         special_token.requires_grad_(True)
#         self.register_buffer('special_token', special_token)

#     def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
#         with torch.no_grad():
#             # assert self.special_token not in src_tokens
#             B = src_tokens.size(0)
#             special_tokens = repeat(self.special_token, f'd -> {B} d')
#             src_tokens = torch.cat((special_tokens, src_tokens), dim=1)
#             src_seq_len += 1
#             first += (first > 0)
#             # FIXME: first at padding
#         return super().forward(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first)
