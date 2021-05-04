import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file_path", default="subsampled", help="Source Translation file path")
    parser.add_argument("--tgt_file_path", default=None, help="Target Transalted file path")
    parser.add_argument("--use_cpu", default=None, help="GPU or CPU Training")
    parser.add_argument("--batch_size", default=32, type=int, help='batch size for the training')
    parser.add_argument("--steps", default=1000, type=int, help='Steps to be executed')
    parser.add_argument("--logger", default=None, help='Either neptune or wandb')
    parser.add_argument("--src_lang", default=None, help='Provide the source language')
    parser.add_argument("--tgt_lang", default=None, help='Provide the target language')
    return parser