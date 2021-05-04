import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from torch import Tensor
import re



class Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

def scaled_dot_product_attention(query:Tensor, key:Tensor, value:Tensor) -> Tensor :
    '''
    Input:
        query = <X, W_q> => (L, d_k)
        key = <X, W_k> => (L, d_k)
        value = <X, W_k> => (L, d_v)
    Returns:
        Self Attention Matrix (L, d_v)
    '''
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

def positional_encoding(seq_len, d_embedding, device=torch.device('cpu')):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1,-1,1)
    dim = torch.arange(d_embedding, dtype=torch.float, device=device).reshape(1,1,-1)
    phase = pos / (1e4 ** (dim // d_embedding))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase) )


def feed_forward(dim_embedding=512, dim_feedforward=2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_embedding, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_embedding)
    )


def preprocess(sentence):
    sentence = sentence.strip()
    sentence = re.sub(
    r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
    sentence = re.sub(r"[ ]+", " ", sentence)
    sentence = re.sub(r"\!+", "!", sentence)
    sentence = re.sub(r"\,+", ",", sentence)
    sentence = re.sub(r"\?+", "?", sentence)
    sentence = sentence.lower()
    return sentence
