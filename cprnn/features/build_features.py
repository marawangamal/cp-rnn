import logging
import os
import sys
import os.path as osp
import argparse as argparse

import torch
import torchtext

from cprnn.features.tokenizer import CharacterTokenizer
from cprnn.utils import save_object

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# todo: fix imports with setup
data_path = {
    "raw": osp.join(sys.path[1], "data", "raw"),
    "processed": osp.join(sys.path[1], "data", "processed"),
}


def ptb_get_indices(dataset: torch.utils.data.Dataset):
    """Merge whole dataset/corpus into single vector"""
    dataset_ids = torch.zeros(sum([len(line) for line in dataset]), dtype=torch.int32)
    tokenizer = CharacterTokenizer()

    idx = 0
    for line in dataset:
        for char in line:
            dataset_ids[idx] = tokenizer.tokenize(char)
            idx += 1

    return dataset_ids, tokenizer


def ptb_make_dataset(**kwargs):
    logger.info("Processing Penn Treebank dataset...")
    for split in ['test', 'valid']:
        dataset = torchtext.datasets.PennTreebank(root=osp.join(data_path['raw'], 'ptb', split), split=split)
        dataset_ids, tokenizer = ptb_get_indices(dataset)

        if not osp.exists(osp.join(data_path['processed'], 'ptb')):
            os.makedirs(osp.join(data_path['processed'], 'ptb'))
        torch.save(dataset_ids, osp.join(data_path['processed'], 'ptb', split + '.pth'))
        save_object(tokenizer, osp.join(data_path['processed'], 'ptb', 'tokenizer.pkl'))

    logging.info("Done. Files saved to {}".format(osp.join(data_path['processed'], 'ptb', split + '.pth')))


make_dataset_functions = {
    'ptb': ptb_make_dataset
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build datasets for language modelling')
    parser.add_argument('-d', '--dataset', type=str, default='ptb', choices=make_dataset_functions.keys())
    args = parser.parse_args()

    make_dataset_functions[args.dataset](**vars(args))




