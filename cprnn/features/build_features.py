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
from cprnn.models import LSTM, CPLSTM, CPRNN, SecondOrderRNN, SecondOrderRNNKR

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
data_path = {
    "raw": osp.join(osp.split(osp.split(ROOT_DIR)[0])[0], "data", "raw"),
    "processed": osp.join(osp.split(osp.split(ROOT_DIR)[0])[0], "data", "processed")
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
    import pdb; pdb.set_trace()
    for split in ['train']:
        dataset = torchtext.datasets.PennTreebank(root=osp.join(data_path['raw'], 'ptb', split), split=split)
        dataset_ids, tokenizer = ptb_get_indices(dataset)

        if not osp.exists(osp.join(data_path['processed'], 'ptb')):
            os.makedirs(osp.join(data_path['processed'], 'ptb'))
        torch.save(dataset_ids, osp.join(data_path['processed'], 'ptb', split + '.pth'))
        save_object(tokenizer.tokens, osp.join(data_path['processed'], 'ptb', 'tokenizer.pkl'))

    logging.info("Done. Files saved to {}".format(osp.join(data_path['processed'], 'ptb', split + '.pth')))


def toy_make_dataset(input_size=32, hidden_size=32, vocab_size=16, rank=32, train_length=1000, valid_length=100,
                     test_length=100, model_name='2rnnkr', **kwargs):
    """Creates toy dataset from RNN model."""
    logger.info("Creating toy dataset...")

    generator_name = 'toy-{}-i{}-h{}-v{}-r{}'.format(model_name, input_size, hidden_size, vocab_size, rank)
    if not osp.exists(osp.join(data_path['processed'], generator_name)):
        os.makedirs(osp.join(data_path['processed'], generator_name))

    # torch.manual_seed(87139)
    models = {"cprnn": CPRNN, "cplstm": CPLSTM, "lstm": LSTM, "2rnnkr": SecondOrderRNNKR}

    batch_size, sequence_length = 1, 1  # can't change these

    model = models[model_name.lower()](input_size=input_size, hidden_size=hidden_size, vocab_size=vocab_size, rank=rank)

    model.eval()
    tokenizer = CharacterTokenizer(tokens=[s for s in string.printable[:vocab_size]])

    for split, dataset_length in zip(['train', 'valid', 'test'], [train_length, valid_length, test_length]):
        init_id = torch.randint(1, vocab_size, (sequence_length, batch_size))
        dataset_ids = list([init_id.item()])

        h_t_prev = torch.zeros(batch_size, model.hidden_size).to(init_id.device)
        for _ in range(dataset_length):
            output_id, output_conf, h_t = model(init_id, h_t_prev)  # [S, B, D_i]
            dataset_ids.append(output_id.item())
            init_id, h_t_prev = output_id, h_t

        print("Split: {}".format(split))
        print(dataset_ids)

        torch.save(torch.tensor(dataset_ids), osp.join(data_path['processed'], generator_name, split + '.pth'))
        save_object(tokenizer.tokens, osp.join(data_path['processed'], generator_name, 'tokenizer.pkl'))

    logging.info("Done. Files saved to {}. Train/Valid/Test length = {}/{}/{}".format(
        osp.join(data_path['processed'], generator_name), train_length, valid_length, test_length)
    )


make_dataset_functions = {
    'ptb': ptb_make_dataset,
    'toy': toy_make_dataset
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build datasets for language modelling')
    parser.add_argument('-d', '--dataset', type=str, default='ptb', choices=make_dataset_functions.keys())
    args = parser.parse_args()

    make_dataset_functions[args.dataset](**vars(args))
