import os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *
from ArgParser import get_parser
from sklearn.model_selection import train_test_split
from create_dataset import *
import spacy
from model.Transformer import Transformer


# load the parser
parser = get_parser()
args = parser.parse_args()

# Set up the Transformer config
config = Config({
    "d_embedding": 512,
    "heads": 8,
    "layers": 6,
    "batch_size": args.batch_size,
    "src_file_path": args.src_file_path,
    "tgt_file_path": args.tgt_file_path,
    "src_lang": args.src_lang,
    "tgt_lang": args.tgt_lang,
    "device": 'cuda:0' if not args.use_cpu else "cpu",
    "steps": args.steps,
})


# Build the dataset
src_nlp = spacy.load("en_core_web_sm")
tgt_nlp = spacy.load("fr_core_news_sm")

src_vocab = build_vocab(config.src_file_path, src_nlp.tokenizer)
trg_vocab = build_vocab(config.tgt_file_path, tgt_nlp.tokenizer)

# print(f'source dictionary:{src_vocab}, target dictionary:{trg_vocab}')

src_special_tokens = Config({
    "bos_idx": src_vocab.stoi["<bos>"],
    "eos_idx": src_vocab.stoi["<eos>"],
    "pad_idx": src_vocab.stoi["<pad>"]
})

tgt_special_tokens = Config({
    "bos_idx": trg_vocab.stoi["<bos>"],
    "eos_idx": trg_vocab.stoi["<eos>"],
    "pad_idx": trg_vocab.stoi["<pad>"]
})

generate_batch = GenerateBatch(src_special_tokens, tgt_special_tokens)

dataset = data_process(config.src_file_path, config.tgt_file_path, src_nlp.tokenizer, tgt_nlp.tokenizer, src_vocab, trg_vocab)
dataset = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=generate_batch)
print(next(iter(dataset)))
sys.exit(0)
train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=True, collate_fn=generate_batch)



if __name__ == '__main__':
    # Intialize the model parameters
    model = Transformer()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    pl.seed_everything(42)
    trainer = pl.Trainer(auto_scale_batch_size='power', deterministic=True)
    trainer.fit(model, DataLoader(dataset))


'''
python --src_lang en --trg_lang fr --src_file_path data/english.txt --trg_file_path data/french.txt
'''

