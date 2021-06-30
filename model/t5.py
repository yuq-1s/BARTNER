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
    def __init__(self, decoder, pad_token_id, label_ids):
        super().__init__()
        # assert isinstance(decoder, T5Stack)
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        mapping = torch.LongTensor([EOS_ID]+label_ids) # T5 has no bos; eos is 1
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个

        logging.warning("FIXME: This version of T5Decoder forces `first` to be None.")

    def forward(self, tokens, state):
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(PAD_ID).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
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
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                past_key_values=past_key_values,
                                # use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        # eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[EOS_ID:EOS_ID+1])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_end_id:self.label_end_id+1])  # bsz x max_len x 1

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

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


class OldT5Seq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False, use_prompt=False):
        model = T5Model.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder

        __normalize = lambda x: (x - x.mean()) / x.std() * 0.4356 - 0.0094

        _tokenizer = T5Tokenizer.from_pretrained(bart_model)
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


        encoder = FT5Encoder(encoder)
        if decoder_type is None:
            decoder = FT5Decoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
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

class T5Seq2SeqModel(OldT5Seq2SeqModel):
    pass
    # def __init__(self, encoder, decoder):
    #     super().__init__(encoder, decoder)
    #     self.special_token = T5Tokenizer.from_pretrained('facebook/bart-large').get_vocab()['ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤ']

    # def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
    #     # self._tokenizer
    #     with torch.no_grad():
    #         assert self.special_token not in src_tokens
    #         B = src_tokens.size(0)
    #         zeros = torch.zeros(B, 1, device='cuda', dtype=int)
    #         special_token = torch.ones(B, 1, device='cuda', dtype=int) * self.special_token
    #         src_tokens = torch.cat((zeros, special_token, src_tokens[:, 1:]), dim=1)
    #         src_seq_len += 1
    #         first += (first > 0)
    #         # FIXME: first at padding
    #     return super().forward(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first)