import os
import torch
import pytorch_lightning as pl
from torch.nn import functional as f
import pandas as pd
from torchtext import data
import spacy
import re, io
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from utils.Helper import preprocess
from torch.nn.utils.rnn import pad_sequence


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding='utf-8') as f:
        for _string in f:
            _string = preprocess(_string)
            counter.update(tokenizer(_string))
    print(f"Dictionary: {counter.keys()}")
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def data_process(src_path, tgt_path, src_tokenizer, trg_tokenizer, src_vocab, trg_vocab):
    src_iter = iter(io.open(src_path, encoding="utf8"))
    trg_iter = iter(io.open(tgt_path, encoding="utf8"))
    data = []
    for raw_src, raw_trg in zip(src_iter, trg_iter):
        raw_src, raw_trg = preprocess(raw_src), preprocess(raw_trg)
        # print(f"src:{raw_src}, target: {raw_trg}")
        src_tensor = torch.tensor([src_vocab[token] for token in src_tokenizer(raw_src)], dtype=torch.long)
        trg_tensor = torch.tensor([trg_vocab[token] for token in trg_tokenizer(raw_trg)], dtype=torch.long)

        data.append((src_tensor, trg_tensor))
    # print(f"data: {data}")
    return data


class GenerateBatch(object):
    def __init__(self, *params):
        self.src_special_tokens,  self.tgt_special_tokens = params
        print(self.src_special_tokens)
    
    def __call__(self, data_batch):
        src_batch, tgt_batch = [], []
        for src_item, tgt_item in data_batch:
            src_batch.append(
                torch.cat([
                    torch.tensor([self.src_special_tokens.bos_idx]), src_item, torch.tensor([self.src_special_tokens.eos_idx])], 
                    dim=0
                )
            )
            tgt_batch.append(
                torch.cat([
                    torch.tensor([self.tgt_special_tokens.bos_idx]), tgt_item, torch.tensor([self.tgt_special_tokens.eos_idx])], 
                    dim=0
                )
            )
        src_batch = pad_sequence(src_batch, padding_value=self.src_special_tokens.pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.tgt_special_tokens.pad_idx)
        return src_batch, tgt_batch

# def generate_batch(data_batch, src_pad_idx, src_bos_idx, src_eos_idx, tgt_pad_idx, tgt_bos_idx, tgt_eos_idx):
#     src_batch, tgt_batch = [], []
#     for src_item, tgt_item in data_batch:
#         src_batch.append(torch.cat([torch.tensor([src_bos_idx]), src_item, [torch.tensor([src_eos_idx])]], dim=0))
#         tgt_batch.append(torch.cat([torch.tensor([tgt_bos_idx]), tgt_item, [torch.tensor([tgt_eos_idx])]], dim=0))
#     src_batch = pad_sequence(src_batch, padding_value=src_pad_idx)
#     tgt_batch = pad_sequence(tgt_batcg, padding_value=tgt_pad_idx)
#     return src_batch, tgt_batch