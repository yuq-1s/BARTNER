import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from data.pipe import BartNERPipe
from model.t5 import T5Seq2SeqModel
import fitlog

from fastNLP import Trainer, Tester
from model.metrics import Seq2SeqSpanMetric
from model.losses import Seq2SeqLoss, T5Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results

from model.callbacks import WarmupCallback
from fastNLP.core.sampler import SortedSampler
from model.generater import SequenceGeneratorModel
from fastNLP.core.sampler import ConstantTokenNumSampler as ConstTokenNumSampler
from model.callbacks import FitlogCallback, SaveEveryEpochCallback

fitlog.debug()
fitlog.set_log_dir('logs')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='conll2003', type=str)

args= parser.parse_args()
dataset_name = args.dataset_name
args.length_penalty = 1
args.save_model = 1

# word: 生成word的start; bpe: 生成所有的bpe; span: 每一段按照start end生成; span_bpe: 每一段都是start的所有bpe，end的所有bpe
args.target_type = 'word'
# args.bart_name = 'facebook/bart-base'
# args.bart_name = 't5-large'
args.bart_name = 't5-11b'
args.schedule = 'linear'
args.decoder_type = None # 'avg_feature'
args.n_epochs = 300
args.num_beams = 1
if args.bart_name == 't5-base':
    args.adapter_size = 192
    args.batch_size = 64
    args.dev_batch_size = 64
elif args.bart_name == 't5-large':
    args.adapter_size = 192*4
    args.batch_size = 24
    args.dev_batch_size = 24
elif args.bart_name == 't5-3b':
    args.adapter_size = 192*4
    args.batch_size = 16
    args.dev_batch_size = 16
elif args.bart_name == 't5-11b':
    args.adapter_size = 192*4
    args.batch_size = 4
    args.dev_batch_size = 4
args.use_encoder_mlp = 1
args.lr = 3e-3
args.warmup_ratio = 0.01
args.mode = 'adapter'
args.do_train = True
args.checkpoint_path = None # 'ckpts/t5-large_adapter_0.001_crossattn_adapter_truncate_decoded/latest_SequenceGeneratorModel_f_2021-07-23-13-43-26-824845'
eval_start_epoch = 0

# the following hyper-parameters are for target_type=word
if dataset_name == 'conll2003':  # three runs get 93.18/93.18/93.36 F1
    max_len, max_len_a = 10, 0.6
elif dataset_name == 'en-ontonotes':  # three runs get 90.46/90.4/90/52 F1
    max_len, max_len_a = 10, 0.8
elif dataset_name == 'CADEC':
    max_len, max_len_a = 10, 1.6
    args.num_beams = 4
    args.lr = 2e-5
    args.n_epochs = 30
    eval_start_epoch=10
elif dataset_name == 'Share_2013':
    max_len, max_len_a = 10, 0.6
    args.use_encoder_mlp = 0
    args.num_beams = 4
    args.lr = 2e-5
    eval_start_epoch = 5
elif dataset_name == 'Share_2014':
    max_len, max_len_a = 10, 0.6
    args.num_beams = 4
    eval_start_epoch = 5
    args.n_epochs = 30
elif dataset_name == 'genia':  # three runs: 79.29/79.13/78.75
    max_len, max_len_a = 10, 0.5
    args.target_type = 'span'
    args.lr = 2e-5
    args.warmup_ratio = 0.01
elif dataset_name == 'en_ace04':  # four runs: 86.84/86.33/87/87.17
    max_len, max_len_a = 50, 1.1
    args.lr = 4e-5
elif dataset_name == 'en_ace05':  # three runs: 85.39/84.54/84.75
    max_len, max_len_a = 50, 0.7
    args.lr = 3e-5
    args.batch_size = 12
    args.num_beams = 4
    args.warmup_ratio = 0.1


save_model = args.save_model
del args.save_model
lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams

length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
target_type = args.target_type
bart_name = args.bart_name
schedule = args.schedule
use_encoder_mlp = args.use_encoder_mlp

fitlog.add_hyper(args)

#######hyper
#######hyper

demo = False
if demo:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{target_type}_demo.pt"
else:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{target_type}.pt"

@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartNERPipe(tokenizer=bart_name, dataset_name=dataset_name, target_type=target_type)
    data_dir = 'data'
    if dataset_name == 'conll2003':
        # paths = {'test': f"{data_dir}/conll2003/test.txt",
        #          'train': f"{data_dir}/conll2003/train.txt",
        #          'dev': f"{data_dir}/conll2003/dev.txt"}
        paths = f"{data_dir}/conll2003"
        data_bundle = pipe.process_from_file(paths, demo=demo)
    elif dataset_name == 'en-ontonotes':
        paths = f'{data_dir}/en-ontonotes/english'
        data_bundle = pipe.process_from_file(paths)
    else:
        data_bundle = pipe.process_from_file(f'{data_dir}/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

print(f'max_len_a:{max_len_a}, max_len:{max_len}')

print(data_bundle)
print("The number of tokens in tokenizer ", len(tokenizer.get_vocab()))

bos_token_id = tokenizer.pad_token_id
eos_token_id = 0 # This is not `tokenizer.eos_token_id`, but the model.decoder.mapping.index(tokenizer.eos_token_id)
label_ids = list(mapping2id.values())
model = T5Seq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                   use_encoder_mlp=use_encoder_mlp, use_prompt=('prompt' in args.mode),
                                   use_adapter=('adapter' in args.mode),
                                   checkpoint_path=args.checkpoint_path,
                                #    checkpoint_path='ckpts/t5-large_adapter+prompt_0.001_decoder_type_none_no_encoder_mlp_normalize_embed/latest_SequenceGeneratorModel_f_2021-07-17-23-57-44-082462',
                                #    checkpoint_path='t5-11b_finetune_decoder_type_none_no_encoder_mlp_normalize_embed/best_SequenceGeneratorModel_f_2021-07-10-10-13-33-882992' if args.mode == 'test' else None
                                #    checkpoint_path='t5_base_decoder_type_none_no_encoder_mlp_normalize_embed1/best_SequenceGeneratorModel_f_2021-07-02-12-20-09-872950' if args.mode == 'test' else None,
                                   model_parallel=(bart_name in ['t5-11b', 't5-3b']),
                                   adapter_size=args.adapter_size,
)

vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None)

import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if args.mode == 'full_vocab':
    parameters = [{'lr': lr, 'weight_decay': 1e-2, 'params': []}]
    for name, param in model.named_parameters():
        if name == 'seq2seq_model.encoder.t5_encoder.embed_tokens.weight' or (args.use_encoder_mlp and 'encoder_mlp' in name):
            parameters[0]['params'].append(param)
        else:
            param.requires_grad = False
elif args.mode == 'prompt':
    parameters = [{'lr': lr, 'weight_decay': 1e-2, 'params': []}]
    for name, param in model.named_parameters():
        if name == 'seq2seq_model.encoder.soft_prompt_embed.weight' or (args.use_encoder_mlp and 'encoder_mlp' in name):
            parameters[0]['params'].append(param)
        else:
            param.requires_grad = False
elif args.mode == 'finetune':
    parameters = []
    params = {'lr':lr, 'weight_decay':1e-2}
    params['params'] = [param for name, param in model.named_parameters() if not ('t5_encoder' in name or 'decoder' in name)]
    if params['params']:
        parameters.append(params)

    params = {'lr':lr, 'weight_decay':1e-2}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('t5_encoder' in name or 'decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    if params['params']:
        parameters.append(params)

    params = {'lr':lr, 'weight_decay':0}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('t5_encoder' in name or 'decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    if params['params']:
        parameters.append(params)
elif args.mode == 'adapter':
    parameters = [{'lr': lr, 'weight_decay': 1e-2, 'params': []}]
    for name, param in model.named_parameters():
        if 'adapter' in name or (args.use_encoder_mlp and 'encoder_mlp' in name) or \
            ('layernorm' in name or 'layer_norm' in name):
            parameters[0]['params'].append(param)
        else:
            param.requires_grad = False
elif args.mode == 'adapter+prompt':
    parameters = [{'lr': lr, 'weight_decay': 1e-2, 'params': []}]
    for name, param in model.named_parameters():
        if name == 'seq2seq_model.encoder.soft_prompt_embed.weight' or 'adapter' in name or (args.use_encoder_mlp and 'encoder_mlp' in name):
            parameters[0]['params'].append(param)
        else:
            param.requires_grad = False
else:
    raise ValueError(f"Unknown mode {args.mode}")

def trainable_names(model, parameters):
    names, num_param = [], 0
    for name, param in model.named_parameters():
        if any(param is p for p in parameters[0]['params']):
            names.append(name)
            num_param += param.numel()
    return names, num_param

trainable_param_names, num_param = list(trainable_names(model, parameters))
print(f"Trainable parameters: {len(trainable_param_names), num_param}")

if args.do_train:
    optimizer = optim.AdamW(parameters)

callbacks = []
callbacks.append(GradientClipCallback(clip_value=0.1, clip_type='value'))
callbacks.append(WarmupCallback(warmup=args.warmup_ratio, schedule=schedule))
# callbacks.append(SaveEveryEpochCallback())

from fastNLP.core.callback import Callback
class DebugCallback(Callback):
    def on_backward_begin(self, loss):
        if loss.item() < 3:
            import pdb; pdb.set_trace()

# callbacks.append(DebugCallback())

if dataset_name not in ('conll2003', 'genia'):
    callbacks.append(FitlogCallback(data_bundle.get_dataset('test'), raise_threshold=-1, # 0.04,
                                        eval_begin_epoch=eval_start_epoch))  # 如果低于0.04大概率是讯飞了
    eval_dataset = data_bundle.get_dataset('dev')
elif dataset_name == 'genia':
    dev_indices = []
    tr_indices = []
    for i in range(len(data_bundle.get_dataset('train'))):
        if i%4==0 and len(dev_indices)<1669:
            dev_indices.append(i)
        else:
            tr_indices.append(i)
    eval_dataset = data_bundle.get_dataset('train')[dev_indices]
    data_bundle.set_dataset(data_bundle.get_dataset('train')[tr_indices], name='train')
    print(data_bundle)
    callbacks.append(FitlogCallback(data_bundle.get_dataset('test'), raise_threshold=0.04, eval_begin_epoch=eval_start_epoch))  # 如果低于0.04大概率是讯飞了
    fitlog.add_other(name='demo', value='split dev')
else:
    callbacks.append(FitlogCallback(raise_threshold=0.04, eval_begin_epoch=eval_start_epoch))  # 如果低于0.04大概率是讯飞了
    eval_dataset = data_bundle.get_dataset('test')

sampler = None
if dataset_name in ('Share_2013',) :
    if target_type == 'bpe':
        sampler = ConstTokenNumSampler('src_seq_len', max_token=3500)
    else:
        sampler = ConstTokenNumSampler('src_seq_len', max_token=4000)
if dataset_name in ('en_ace04',) and target_type == 'bpe':
    sampler = ConstTokenNumSampler('src_seq_len', max_sentence=batch_size, max_token=2500)
elif ('large' in bart_name and dataset_name in ('en-ontonotes', 'genia')):
    sampler = ConstTokenNumSampler('src_seq_len', max_token=3000)
else:
    sampler = BucketSampler(seq_len_field_name='src_seq_len')

metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), target_type=target_type, has_bos=False)

ds = data_bundle.get_dataset('train')
if dataset_name == 'conll2003':
    # FIXME: ds has no method `concat`
    # ds.concat(data_bundle.get_dataset('dev'))
    data_bundle.delete_dataset('dev')
if save_model == 1:
    save_path = f'ckpts/{args.bart_name}_{args.mode}_{args.lr}_{args.adapter_size}_no_dup_layernorm/'
else:
    save_path = None

if not args.do_train:
    tester = Tester(eval_dataset, model, metrics=metric, device=device, callbacks=callbacks, batch_size=args.dev_batch_size)
    tester.test()
    import sys; sys.exit(0)

validate_every = 40000 // args.batch_size
eval_dataset = eval_dataset[:2048]
print(f"#param = {sum(p.numel() for p in model.parameters())}")
trainer = Trainer(train_data=ds, model=model, optimizer=optimizer,
# trainer = Trainer(train_data=eval_dataset, model=model, optimizer=optimizer,
                  loss=T5Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=4, n_epochs=n_epochs, print_every=1 if 'SEARCH_OUTPUT_FP' not in os.environ else 100,
                  dev_data=eval_dataset, metrics=metric, metric_key='f',
                  validate_every=validate_every, save_path=save_path, use_tqdm='SEARCH_OUTPUT_FP' not in os.environ, device=device,
                  callbacks=callbacks, check_code_level=0, test_use_tqdm='SEARCH_OUTPUT_FP' not in os.environ,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=args.dev_batch_size)

trainer.train(load_best_model=True)

