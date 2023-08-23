from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
import datasets
import json
import copy
import subprocess
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
keys to set:
batch_size in mask_examples
batch_size in tokenizer and Dataloader should be the same (padding)
"""
bs_mask = 128
bs_model = 16


def mask_examples(examples):
    global src_dataset, cnt
    mask_sents = []
    labels = []
    for sent in examples['text']:
        src_sent = src_dataset[cnt]['text'].strip()
        cnt += 1
        words = sent.strip().split(" ")
        for idx_wd in range(len(words)):
            seq = copy.deepcopy(words)
            seq[idx_wd] = mask
            # sequence = " ".join(seq)  # 什么都不加

            # seq.append(sep)
            # sequence = " ".join(seq) + " " + src_sent  # 后缀

            seq.insert(0, sep)
            sequence = src_sent + " " + " ".join(seq)  # 前缀
            # sequence = src_sent + " " + " ".join(seq) + " " + src_sent  # 前后都加 y3 y4 y3
            # sequence = src_sent + " translate " + " ".join(seq) + " translate " + src_sent
            mask_sents.append(sequence)
            if idx_wd == len(words) - 1:
                labels.append(1)
            else:
                labels.append(0)
    return {'mask_sents': mask_sents, 'labels': labels}


def tokenize_function(examples):
    # return tokenizer(example['text'], truncation=True, padding=True, max_length=512, return_tensors='pt')
    return tokenizer(examples['mask_sents'], padding=True, truncation=True, max_length=512,
                     add_special_tokens=True, return_tensors='pt')

# model_pth = "F:/project/UNMT/LM/XLMR"
# input_pth = "F:/project/UNMT/LM/MUSE/results/translate/ro2en_w"
# gt_pth = "F:/project/UNMT/LM/MUSE/results/translate/europarl-v8.ro-en.ro"
# output_pth = "F:/project/UNMT/LM/MUSE/results/translate/xlmr_ro2en_w_gt.1.new"
# cache_dir = "F:/project/UNMT/LM/MUSE/cache"
# metric_pth = "F:/project/UNMT/LM/MUSE/cache/sacrebleu"
# checkpoint_pth = "F:/project/UNMT/LM/MUSE/results/translate/checkpoints"

# model_pth = '/dat01/laizhiquan/psl/LM/XLMR-base/'
# input_pth = '/dat01/laizhiquan/psl/Project/UNMT/MUSE/results/translate/en-ro/wiki/en2ro_w'
# gt_pth = '/dat01/laizhiquan/psl/Project/UNMT/MUSE/results/translate/ground_truth/europarl-v8.ro-en.en'
# output_pth = '/dat01/laizhiquan/psl/Project/UNMT/MUSE/results/translate/en-ro/wiki/xlmr_en2ro_src_1mask'
# cache_dir = '/dat01/laizhiquan/psl/Project/UNMT/MUSE/cache'
# metric_pth = "/dat01/laizhiquan/psl/Project/UNMT/MUSE/cache/sacrebleu"
# sacrebleu_path = '/dat01/laizhiquan/psl/Project/UNMT/MUSE/results/translate/en-ro/wiki/sacrebleu_xlmr_en2ro_src_1mask'

# ft_mass
model_pth = '/home/tianzhiliang/dat01/psl/models/xlmr_large/'
input_pth = '/home/tianzhiliang/dat01/psl/scripts/results/de2en/en2kk.hyp'
gt_pth = '/home/tianzhiliang/dat01/psl/scripts/results/de2en/en2kk.ref'
src_pth = '/home/tianzhiliang/dat01/psl/data/kk2en_parallel/test/test.kk_KZ'
output_pth = '/home/tianzhiliang/dat01/psl/scripts/results/de2en/xlmr_prefix_en2kk.hyp'
cache_dir = '/home/tianzhiliang/dat01/psl/cache'
# metric_pth = "/dat01/laizhiquan/psl/Project/UNMT/MUSE/cache/sacrebleu"
# sacrebleu_path = '/dat01/laizhiquan/psl/Project/UNMT/MASS/bleu/ft_mass_xlmr_src-y4-src.1'
# BLEU_SCRIPT_PATH = '/dat01/laizhiquan/psl/Project/UNMT/MASS/src/evaluation/multi-bleu.perl'

# pt_mass
# model_pth = '/dat01/laizhiquan/psl/LM/XLMR-large/'
# input_pth = '/dat01/laizhiquan/psl/Project/UNMT/MASS/dumped/unsupMT_enro/mlm_pt_eval_y4bleu_y4ppl_decy4_enc1_word/hypotheses/hyp0.ro-en.test.y3.txt'
# src_pth = '/dat01/laizhiquan/psl/Project/UNMT/MASS/dumped/unsupMT_enro/mlm_pt_eval_y4bleu_y4ppl_decy4_enc1_word/hypotheses/ref.en-ro.test.txt'
# gt_pth = '/dat01/laizhiquan/psl/Project/UNMT/MASS/dumped/unsupMT_enro/mlm_pt_eval_y4bleu_y4ppl_decy4_enc1_word/hypotheses/ref.ro-en.test.txt'
# output_pth = '/dat01/laizhiquan/psl/Project/UNMT/MASS/dumped/unsupMT_enro/mlm_pt_eval_y4bleu_y4ppl_decy4_enc1_word/hypotheses/hyp0.ro-en.test.y4_xlmr.txt'
# cache_dir = '/dat01/laizhiquan/psl/Project/UNMT/MASS/cache'
# metric_pth = "/dat01/laizhiquan/psl/Project/UNMT/MUSE/cache/sacrebleu"
# sacrebleu_path = '/dat01/laizhiquan/psl/Project/UNMT/MASS/bleu/pt_mass_xlmr_src-y4'
# BLEU_SCRIPT_PATH = '/dat01/laizhiquan/psl/Project/UNMT/MASS/src/evaluation/multi-bleu.perl'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Auto
tokenizer = AutoTokenizer.from_pretrained(model_pth)
model = AutoModelForMaskedLM.from_pretrained(model_pth)
model = model.to(device)

mask = f"{tokenizer.mask_token}"
sep = f"{tokenizer.sep_token}"

raw_test_dataset = datasets.load_dataset('text', cache_dir=cache_dir, data_files={'test': input_pth}, split='test')
src_dataset = datasets.load_dataset('text', cache_dir=cache_dir, data_files={'test': src_pth}, split='test')
gt_dataset = datasets.load_dataset('text', cache_dir=cache_dir, data_files={'test': gt_pth}, split='test')
# metric = datasets.load_metric(metric_pth, cache_dir=cache_dir)

cnt = 0
mask_test_dataset = raw_test_dataset.map(mask_examples, batched=True, batch_size=bs_mask,
                                         remove_columns=raw_test_dataset.column_names)
# mask_test_dataset = raw_test_dataset.map(lambda example, idx: mask_examples(example, idx, gt_dataset),
#                                          with_indices=True, remove_columns=raw_test_dataset.column_names)
# print(mask_test_dataset)


tokenized_mask_test_dataset = mask_test_dataset.map(lambda examples: tokenizer(examples['mask_sents'],
                                                                               padding=True, truncation=True,
                                                                               max_length=512, add_special_tokens=True),
                                                    batched=True, batch_size=bs_model,
                                                    remove_columns=['mask_sents'])
# print(tokenized_mask_test_dataset['input_ids'])

tokenized_mask_test_dataset.set_format('torch')

# tokenized_mask_test_dataset['input_ids'] 中一个就是一个batch
# print([len(x) for x in tokenized_mask_test_dataset['input_ids']])

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# batch = data_collator(tokenized_mask_test_dataset)
# print(batch)
# print([len(x) for x in batch["input_ids"]])

eval_dataloader = DataLoader(tokenized_mask_test_dataset, batch_size=bs_model)
# eval_dataloader = DataLoader(tokenized_mask_test_dataset)
model.eval()
trans_tokens = []
cnt = 0
with open(output_pth, 'w', encoding='utf-8', buffering=1) as o:
    for batch in eval_dataloader:
        # batch["input_ids"] = torch.squeeze(torch.stack(batch["input_ids"], 0)).to(device)
        # batch["attention_mask"] = torch.squeeze(torch.stack(batch["attention_mask"], 0)).to(device)
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch['labels'] = batch['labels'].to(device)
        indexes = (batch['labels'] == 1).nonzero(as_tuple=True)[0].tolist()
        del batch['labels']
        mask_token_index = torch.where(batch["input_ids"] == tokenizer.mask_token_id)[-1]
        with torch.no_grad():
            outputs = model(**batch)
        # outputs = model(**batch)
        token_logits = outputs.logits
        mask_token_logits = token_logits[:, mask_token_index, :]
        mask_token_logits_list = []
        for idx in range(mask_token_logits.shape[0]):
            mask_token_logits_list.append(mask_token_logits[idx, idx, :])
        mask_token_logits = torch.stack(mask_token_logits_list)
        # mask_token_logits = token_logits[0, mask_token_index[0], :]
        # for idx in range(1, bs_model):
        #     torch.stack((mask_token_logits, token_logits[idx, mask_token_index[idx], :]), out=mask_token_logits)
        values, indices = torch.topk(mask_token_logits, 1, dim=1, sorted=True)
        top_1_tokens = torch.squeeze(indices).tolist()
        if indexes:
            del_len = 0
            for index in indexes:
                index -= del_len
                trans_tokens.extend(top_1_tokens[:index+1])
                # trans_sent = tokenizer.decode(trans_tokens)
                trans_sent = tokenizer.decode(trans_tokens, skip_special_tokens=True)
                o.write(trans_sent + '\n')
                # metric.add(prediction=[trans_sent], reference=[src_dataset[cnt]['text']])
                # metric.add(prediction=[trans_sent], reference=[gt_dataset[cnt]['text']])
                cnt += 1
                trans_tokens.clear()
                del_len += len(top_1_tokens[:index+1])
                del top_1_tokens[:index+1]
            trans_tokens.extend(top_1_tokens)
        else:
            trans_tokens.extend(top_1_tokens)
        # tran_sent = tokenizer.decode(top_1_tokens)
        # o.write(tran_sent + '\n')
        # metric.add(prediction=[tran_sent], reference=[gt_dataset[idx]['text']])
    # with open(sacrebleu_path, 'w', encoding='utf-8') as sacre:
    #     sacre.write('sacrebleu:\n')
    #     sacre.write(json.dumps(metric.compute()))
    #     sacre.write('\nmulti-bleu:\n')
    #     command = 'perl ' + BLEU_SCRIPT_PATH + ' %s < %s'
    #     p = subprocess.Popen(command % (gt_pth, output_pth), stdout=subprocess.PIPE, shell=True)
    #     result = p.communicate()[0].decode("utf-8")
    #     sacre.write(result)
# print(metric.compute())
