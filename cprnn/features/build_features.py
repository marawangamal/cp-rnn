import logging
import os
import sys
import string
import os.path as osp
import argparse as argparse

import torch
import torchtext

from cprnn.features.tokenizer import CharacterTokenizer
from cprnn.utils import save_object

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

data_path = {
    "raw": osp.join(sys.path[1], "data", "raw"),
    "processed": osp.join(sys.path[1], "data", "processed"),
}


def ptb_get_indices(dataset: torch.utils.data.Dataset):
    """Merge whole dataset/corpus into single vector"""
    dataset_ids = torch.zeros(sum([len(line) for line in dataset]), dtype=torch.long)
    tokenizer = CharacterTokenizer(tokens=[s for s in string.printable])

    idx = 0
    for line in dataset:
        for char in line:
            dataset_ids[idx] = tokenizer.tokenize(char).item()
            idx += 1

    return dataset_ids, tokenizer


def ptb_make_dataset(**kwargs):
    """Merge whole dataset/corpus into single integer vector and save its tokenizer"""
    logger.info("Processing Penn Treebank dataset...")
    for split in ['train']:
        dataset = torchtext.datasets.PennTreebank(root=osp.join(data_path['raw'], 'ptb', split), split=split)
        dataset_ids, tokenizer = ptb_get_indices(dataset)

        if not osp.exists(osp.join(data_path['processed'], 'ptb')):
            os.makedirs(osp.join(data_path['processed'], 'ptb'))
        torch.save(dataset_ids, osp.join(data_path['processed'], 'ptb', split + '.pth'))
        save_object(tokenizer.tokens, osp.join(data_path['processed'], 'ptb', 'tokenizer.pkl'))

    logging.info("Done. Files saved to {}".format(osp.join(data_path['processed'], 'ptb', split + '.pth')))


def toy_make_dataset(dataset_length=1000, model_name='cprnn', **kwargs):
    """Creates toy dataset from RNN model."""
    logger.info("Creating toy dataset...")

    from cprnn.models import LSTM, CPLSTM, CPRNN
    torch.manual_seed(0)

    models = {"cprnn": CPRNN, "cplstm": CPLSTM, "lstm": LSTM}

    batch_size, sequence_length = 1, 1  # can't change these

    input_size, hidden_size, vocab_size, rank = 8, 8, 4, 4
    model = models[model_name.lower()](input_size=input_size, hidden_size=hidden_size, vocab_size=vocab_size, rank=rank)
    model.eval()

    for split in ['train']:
        init_id = torch.randint(0, vocab_size, (sequence_length, batch_size))
        dataset_ids = list([init_id.item()])

        for _ in range(dataset_length):
            output_id, _ = model(init_id)  # [S, B, D_i]
            dataset_ids.append(output_id)

        if not osp.exists(osp.join(data_path['processed'], 'toy-rnn')):
            os.makedirs(osp.join(data_path['processed'], 'toy-rnn'))
        torch.save(dataset_ids, osp.join(data_path['processed'], 'toy-rnn', split + '.pth'))

    logging.info("Done. Files saved to {}".format(osp.join(data_path['processed'], 'toy-rnn', split + '.pth')))


make_dataset_functions = {
    'ptb': ptb_make_dataset,
    'toy': toy_make_dataset
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build datasets for language modelling')
    parser.add_argument('-d', '--dataset', type=str, default='ptb', choices=make_dataset_functions.keys())
    args = parser.parse_args()

    make_dataset_functions[args.dataset](**vars(args))
