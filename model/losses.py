
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        return loss

class T5Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1)).eq(0)
        tgt_tokens = tgt_tokens.masked_fill(mask, -100)
        tgt_tokens = tgt_tokens.to(pred.device)
        loss = F.cross_entropy(target=tgt_tokens[:, :pred.size(1)], input=pred.transpose(1, 2))
        return loss

class TestingT5Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()
        self._losses = []

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1)).eq(0)
        tgt_tokens = tgt_tokens.masked_fill(mask, -100)
        tgt_tokens = tgt_tokens.to(pred.device)
        loss = F.cross_entropy(target=tgt_tokens[:, :pred.size(1)], input=pred.transpose(1, 2))
        self._losses.append(loss.item())
        if len(self._losses) % 128 == 0:
            print(self.avg_loss_message)
        return loss

    @property
    def avg_loss_message(self):
        return f"{len(self._losses)} step loss avg: {sum(self._losses) / len(self._losses)}"
