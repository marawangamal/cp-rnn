import logging
import os
import sys
import string
import os.path as osp
import argparse as argparse
import urllib.request as urllib2

import torch
import torchtext

from cprnn.features.tokenizer import CharacterTokenizer
from cprnn.utils import save_object
from cprnn.models import CPRNN, LSTM

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = osp.split(osp.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]  # Project Root
data_path = {
    "raw": osp.join(ROOT_DIR, "data", "raw"),
    "processed": osp.join(ROOT_DIR, "data", "processed")
}

models = {"cprnn": CPRNN, "lstm": LSTM}


def torchtext_get_indices(dataset: torch.utils.data.Dataset, level='char'):
    """Merge whole dataset/corpus into single vector"""

    if level == 'char':
        tokenizer = CharacterTokenizer(tokens=[s for s in string.printable])
        dataset_ids = torch.zeros(sum([len(line) for line in dataset]), dtype=torch.long)
    else:
        tokenizer = CharacterTokenizer()
        dataset_ids = torch.zeros(sum([len(line.split()) for line in dataset]), dtype=torch.long)

    idx = 0
    for line in dataset:
        if level == 'char':
            for char in line:
                dataset_ids[idx] = tokenizer.tokenize(char).item()
                idx += 1

        elif level == 'word':
            for word in line.split():
                dataset_ids[idx] = tokenizer.add_token(word.lower())
                idx += 1
        else:
            raise ValueError("Unknown level: {} choose one of [char, word]".format(level))

    return dataset_ids, tokenizer


def torchtext_make_dataset(dataset='ptb', level='char', **kwargs):
    """Merge whole dataset/corpus into single integer vector and save its tokenizer"""
    logger.info("Processing Penn Treebank dataset...")
    for split in ['train', 'valid', 'test']:
        dataset_torchtext = {
            "wikitext103": torchtext.datasets.WikiText103,
            "ptb": torchtext.datasets.PennTreebank
        }[dataset](root=osp.join(data_path['raw'], dataset, split), split=split)
        dataset_ids, tokenizer = torchtext_get_indices(dataset_torchtext, level=level)

        if not osp.exists(osp.join(data_path['processed'], dataset)):
            os.makedirs(osp.join(data_path['processed'], dataset))

        # import pdb; pdb.set_trace()
        torch.save(dataset_ids, osp.join(data_path['processed'], dataset,  '{}-{}.pth'.format(split, level)))

        print("Tokenizer test:")
        print(''.join([tokenizer.ix_to_char(k.item()) for k in dataset_ids[:1000]]))

        if split == 'train':
            save_object(tokenizer.tokens, osp.join(data_path['processed'], dataset, 'tokenizer-{}.pkl'.format(level)))
            logging.info("Tokenizer saved to {} ".format(
                osp.join(data_path['processed'], dataset, 'tokenizer-{}.pkl'.format(level)))
            )

        logging.info("File saved to {} | Length {}".format(
            osp.join(data_path['processed'], dataset,  '{}-{}.pth'.format(split, level)), len(dataset_ids))
        )


def toy_make_dataset(input_size=32, hidden_size=32, vocab_size=16, rank=32, train_length=1000, valid_length=100,
                     test_length=100, model='2rnnkr', **kwargs):
    """Creates toy dataset from RNN model."""
    logger.info("Creating toy dataset using {}".format(model.upper()))

    generator_name = 'toy-{}-i{}-h{}-v{}-r{}'.format(model, input_size, hidden_size, vocab_size, rank)
    if not osp.exists(osp.join(data_path['processed'], generator_name)):
        os.makedirs(osp.join(data_path['processed'], generator_name))

    # torch.manual_seed(87139)

    batch_size, sequence_length = 1, 1  # can't change these

    model = models[model.lower()](input_size=input_size, hidden_size=hidden_size, vocab_size=vocab_size, rank=rank)

    model.eval()
    tokenizer = CharacterTokenizer(tokens=[s for s in string.printable[:vocab_size]])

    for split, dataset_length in zip(['train', 'valid', 'test'], [train_length, valid_length, test_length]):
        input_ids = torch.randint(1, vocab_size, (sequence_length, batch_size))
        dataset_ids = list([input_ids.item()])

        hidden_state_prev = None
        for i in range(dataset_length):
            output_ids, output_conf, hidden_state = model(input_ids, hidden_state_prev)  # [S, B, D_i]
            dataset_ids.append(output_ids.item())
            input_ids, hidden_state_prev = output_ids, hidden_state

        # import pdb; pdb.set_trace()
        hst = torch.bincount(torch.tensor(dataset_ids, dtype=torch.int))
        print("Split: {}\n{}\nHistogram:\n{}\n".format(split, dataset_ids[:10], hst.tolist()))

        torch.save(torch.tensor(dataset_ids), osp.join(data_path['processed'], generator_name, split + '.pth'))
        save_object(tokenizer.tokens, osp.join(data_path['processed'], generator_name, 'tokenizer.pkl'))

    logging.info("Done. Files saved to {}. Train/Valid/Test length = {}/{}/{}".format(
        osp.join(data_path['processed'], generator_name), train_length, valid_length, test_length)
    )


def anna_make_dataset(**kwargs):
    url = "https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/recurrent-neural-networks/char" \
          "-rnn/data/anna.txt "

    dataset_ids = list()
    tokenizer = CharacterTokenizer(tokens=[s for s in string.printable])

    if not osp.exists(osp.join(data_path['raw'], "anna")):
        os.makedirs(osp.join(data_path['raw'], "anna"))

    with open(osp.join(data_path['raw'], "anna", "anna.txt"), "a") as f:
        data = urllib2.urlopen(url)
        logging.info("Processing Anna Karenina dataset...")
        for line in data:  # files are iterable
            dataset_ids.extend([tokenizer.char_to_ix(s) for s in line.decode("utf-8").strip()])
            f.write(line.decode("utf-8"))

    train, valid, test = dataset_ids[:int(0.8 * len(dataset_ids))], \
                         dataset_ids[int(0.8 * len(dataset_ids)):int(0.9 * len(dataset_ids))], \
                         dataset_ids[int(0.9 * len(dataset_ids)):]

    if not osp.exists(osp.join(data_path['processed'], 'anna')):
        os.makedirs(osp.join(data_path['processed'], 'anna'))

    for split, dataset_ids in zip(['train', 'valid', 'test'], [train, valid, test]):
        logging.info("Saving {} split...({} Points)".format(split, len(dataset_ids)))
        torch.save(torch.tensor(dataset_ids), osp.join(data_path['processed'], 'anna', split + '.pth'))
    save_object(tokenizer.tokens, osp.join(data_path['processed'], 'anna', 'tokenizer.pkl'))


make_dataset_functions = {
    'ptb': lambda level, **kwargs: torchtext_make_dataset(level=level, dataset='ptb'),
    'wikitext103': lambda level, **kwargs: torchtext_make_dataset(level=level, dataset='wikitext103'),
    'toy': toy_make_dataset,
    "anna": anna_make_dataset
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build datasets for language modelling')
    parser.add_argument('-d', '--dataset', type=str, default='ptb', choices=make_dataset_functions.keys())
    parser.add_argument('-l', '--level', type=str, default='char', choices=['char', 'word'])
    parser.add_argument('-m', '--model', type=str, default='2rnnkr', choices=models.keys())
    args = parser.parse_args()

    make_dataset_functions[args.dataset](**vars(args))
