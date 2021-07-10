# Towards parameter-efficient tuning of large models for NER

Currently, this repo targets T5 for NER.

## TODO

- [ ] Add MLP, set it as trainable
- [x] Debug T5: SequenceGenerator always 0 f1
- [x] Debug T5: >200 loss on initialization
- [x] Finetune `t5-base` and achieve comparable performance with `bart-large`
- [ ] Finetune `t5-11b` and achieve better performance than `bart-large`
- [ ] Add LM adaption for T5
- [ ] Tuning only MLP after `encoder_out` and `embed_tokens` on `t5-base`
- [ ] Tuning only MLP after `encoder_out` and a soft prompt

## Install

To run this version with model parallel, some modifications to the following libraries are needed.

- fastNLP==0.6.0

```
--- /tmp/trainer_orig.py        2021-07-10 03:57:06.348091151 +0000
+++ .venv/lib/python3.9/site-packages/fastNLP/core/trainer.py       2021-07-10 03:58:34.512642014 +0000
@@ -500,7 +500,11 @@
             raise TypeError("train_data type {} not support".format(type(train_data)))
 
         model.train()
-        self.model = _move_model_to_device(model, device=device)
+        some_param = next(iter(model.parameters()))
+        if some_param.device.type == 'cpu':
+            self.model = _move_model_to_device(model, device=device)
+        else:
+            self.model = model
         if _model_contains_inner_module(self.model):
             self._forward_func = self.model.module.forward
         else:
@@ -789,7 +793,7 @@
         """
         return self.losser(predict, truth)
 
-    def _save_model(self, model, model_name, only_param=False):
+    def _save_model(self, model, model_name, only_param=True):
         r""" 存储不含有显卡信息的state_dict或model
         :param model:
         :param model_name:
```
Otherwise fastNLP will load all parameters into `cuda:0`, invalidating model parallel.

- transformers==4.7.0

```
--- /tmp/modeling_t5_orig.py    2021-07-10 10:32:13.078207891 +0800
+++ .venv/lib/python3.9/site-packages/transformers/models/t5/modeling_t5.py        2021-07-09 16:46:14.836178715 +0800
@@ -244,6 +244,8 @@
         # convert into float16 if necessary
         if self.weight.dtype == torch.float16:
             hidden_states = hidden_states.to(torch.float16)
+        if hidden_states.device != self.weight.device:
+            self.to(hidden_states.device)
         return self.weight * hidden_states
@@ -993,6 +996,8 @@
                     None,  # past_key_value is always None with gradient checkpointing
                 )
             else:
+                if self.model_parallel:
+                    layer_module.to(hidden_states.device)
                 layer_outputs = layer_module(
                     hidden_states,
                     attention_mask=extended_attention_mask,

```

Otherwise torch complains incompatible cuda devices with model parallel.

## Original README

This is the code for ACL-ICJNLP2021 paper [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223).

Install the package in the requirements.txt, then use the following
commands to install two other packages
```text
pip install git+https://github.com/fastnlp/fastNLP@dev
pip install git+https://github.com/fastnlp/fitlog
```

You need to put your data in the parallel folder of this repo
```text
    - BARTNER/
        - train.py
        ...
    - data/
        - conll2003
            - train.txt
            - text.txt
            - dev.txt
        - en-ontonotes
            - ...
        - Share_2013
        - Share_2014
        - CADEC
        - en_ace04
        - en_ace05
        - genia

```
For the `conll2003` and `en-ontonotes` you data in each split should like (The first column is words, the second column is tags. We assume the tag is the BIO-tagging)
```text
LONDON B-LOC
1996-08-30 O

West B-MISC
Indian I-MISC
all-rounder O
Phil B-PER
```

For nested dataset `en_ace04`, `en_ace05` and `genia`, the data should like 
(each line is a jsonline, contains ``ners`` and ``sentences`` keys.)
```text
{"ners": [[[16, 16, "DNA"], [4, 8, "DNA"], [24, 26, "DNA"], [19, 20, "DNA"]], [[31, 31, "DNA"], [2, 2, "DNA"], [4, 4, "DNA"], [30, 31, "DNA"]], [[23, 24, "RNA"], [14, 15, "cell_type"], [1, 2, "RNA"]], [[2, 2, "DNA"]], [], [[0, 0, "DNA"], [9, 9, "cell_type"]]], "sentences": [["There", "is", "a", "single", "methionine", "codon-initiated", "open", "reading", "frame", "of", "1,458", "nt", "in", "frame", "with", "a", "homeobox", "and", "a", "CAX", "repeat", ",", "and", "the", "open", "reading", "frame", "is", "predicted", "to", "encode", "a", "protein", "of", "51,659", "daltons."], ["When", "the", "homeodomain", "from", "HB24", "was", "compared", "to", "known", "mammalian", "and", "Drosophila", "homeodomains", "it", "was", "found", "to", "be", "only", "moderately", "conserved,", "but", "when", "it", "was", "compared", "to", "a", "highly", "diverged", "Drosophila", "homeodomain", ",", "H2.0,", "it", "was", "found", "to", "be", "80%", "identical."], ["The", "HB24", "mRNA", "was", "absent", "or", "present", "at", "low", "levels", "in", "normal", "B", "and", "T", "lymphocytes", ";", "however,", "with", "the", "appropriate", "activation", "signal", "HB24", "mRNA", "was", "induced", "within", "several", "hours", "even", "in", "the", "presence", "of", "cycloheximide", "."], ["Characterization", "of", "HB24", "expression", "in", "lymphoid", "and", "select", "developing", "tissues", "was", "performed", "by", "in", "situ", "hybridization", "."], ["Positive", "hybridization", "was", "found", "in", "thymus", ",", "tonsil", ",", "bone", "marrow", ",", "developing", "vessels", ",", "and", "in", "fetal", "brain", "."], ["HB24", "is", "likely", "to", "have", "an", "important", "role", "in", "lymphocytes", "as", "well", "as", "in", "certain", "developing", "tissues", "."]]}
{"ners": [[[16, 16, "DNA"], [4, 8, "DNA"], [24, 26, "DNA"], [19, 20, "DNA"]], [[31, 31, "DNA"], [2, 2, "DNA"], [4, 4, "DNA"], [30, 31, "DNA"]], [[23, 24, "RNA"], [14, 15, "cell_type"], [1, 2, "RNA"]], [[2, 2, "DNA"]], [], [[0, 0, "DNA"], [9, 9, "cell_type"]]], "sentences": [["There", "is", "a", "single", "methionine", "codon-initiated", "open", "reading", "frame", "of", "1,458", "nt", "in", "frame", "with", "a", "homeobox", "and", "a", "CAX", "repeat", ",", "and", "the", "open", "reading", "frame", "is", "predicted", "to", "encode", "a", "protein", "of", "51,659", "daltons."], ["When", "the", "homeodomain", "from", "HB24", "was", "compared", "to", "known", "mammalian", "and", "Drosophila", "homeodomains", "it", "was", "found", "to", "be", "only", "moderately", "conserved,", "but", "when", "it", "was", "compared", "to", "a", "highly", "diverged", "Drosophila", "homeodomain", ",", "H2.0,", "it", "was", "found", "to", "be", "80%", "identical."], ["The", "HB24", "mRNA", "was", "absent", "or", "present", "at", "low", "levels", "in", "normal", "B", "and", "T", "lymphocytes", ";", "however,", "with", "the", "appropriate", "activation", "signal", "HB24", "mRNA", "was", "induced", "within", "several", "hours", "even", "in", "the", "presence", "of", "cycloheximide", "."], ["Characterization", "of", "HB24", "expression", "in", "lymphoid", "and", "select", "developing", "tissues", "was", "performed", "by", "in", "situ", "hybridization", "."], ["Positive", "hybridization", "was", "found", "in", "thymus", ",", "tonsil", ",", "bone", "marrow", ",", "developing", "vessels", ",", "and", "in", "fetal", "brain", "."], ["HB24", "is", "likely", "to", "have", "an", "important", "role", "in", "lymphocytes", "as", "well", "as", "in", "certain", "developing", "tissues", "."]]}
...
```

For discontinuous dataset `Share_2013`, `Share_2014` and `CADEC`, the data should like (
each sample has two lines, if the second line is empty means there is not entity.
)
```text
Abdominal cramps , flatulence , gas , bloating .
0,1 ADR|3,3 ADR|7,7 ADR|5,5 ADR

Cramps would start within 15 minutes of taking pill , even during meals .
0,0 ADR

...
```
We use code from https://github.com/daixiangau/acl2020-transition-discontinuous-ner to pre-process
 the data.

You can run the code by directly using
```shell
python train.py
```

The following output should be achieved
```text
Save cache to caches/data_facebook/bart-large_conll2003_word.pt.                                                                                                        
max_len_a:0.6, max_len:10
In total 3 datasets:
        test has 3453 instances.
        train has 14041 instances.
        dev has 3250 instances.

The number of tokens in tokenizer  50265
50269 50274
input fields after batch(if batch size is 2):
        tgt_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 8]) 
        src_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 11]) 
        first: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 11]) 
        src_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
        tgt_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
target fields after batch(if batch size is 2):
        entities: (1)type:numpy.ndarray (2)dtype:object, (3)shape:(2,) 
        tgt_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 8]) 
        target_span: (1)type:numpy.ndarray (2)dtype:object, (3)shape:(2,) 
        tgt_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 

training epochs started 2021-06-02-11-49-26-964889
Epoch 1/30:   0%|                                                         | 15/32430 [00:06<3:12:37,  2.80it/s, loss:6.96158
```

Some important python files are listed below
```text
- BartNER
  - data
     - pipe.py # load and process data
  - model
     - bart.py # the model file
  - train.py  # the training file
```

The different ``Loader``s  in the `data/pipe.py` is meant to load data, and the ``data.BartNERPipe`` class 
is to process data, the loader should load data into a DataBundle object,
you can mock the provided Loader to write your own loader, as long as your
dataset has the following four fields, the ``BartNERPipe`` should be able to 
process it
```text
- raw_words  # List[str]
    # ['AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06']
- entities  # List[List[str]]
    # [['AL-AIN'], ['United', 'Arab', 'Emirates']]
- entity_tags  # List[str], the same length as entities
    # ['loc', 'loc']
- entity_spans # List[List[int]], the inner list must have an even number of ints, means the start(inclusive，开区间) and end(exclusive，开区间) of an entity segment
    # [[0, 1], [2, 5]] or for discontinous NER [[0, 1, 5, 7], [2, 3, 5, 7],...]
```

In order to help you reproduce the results, we have hardcoded the hyper-parameters
 for each dataset in the code, you can change them based on your need. 
We conduct all experiments in NVIDIA-3090(24G memory). Some known
 difficulties about the reproduction of this code: (1) Some datasets
(nested and discontinous) will drop to 0 or near 0 F1 during training, please drop these
 results; (2) randomness will cause large performance variance for some datasets, please try to 
run multiple times. 

We deeply understand how frustrating it can be 
if the results are hard to reproduce, we tried our best to make sure 
the results were at least reproducible in our equipment (Usually take 
average from at least  five runs).
