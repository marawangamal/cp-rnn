import argparse as argparse

import torch
import torch.nn as nn

from cprnn.utils import load_object
from cprnn.models.lstm import LSTM
from cprnn.features.ptb_dataloader import PTBDataloader


def main(args):
    import pdb; pdb.set_trace()
    input_size = 128
    hidden_size = 256

    # Offline data processing
    train_dataloader = PTBDataloader('data/processed/ptb/train.pth', batch_size=args.batch_size, seq_len=args.seq_len)
    valid_dataloader = PTBDataloader('data/processed/ptb/valid.pth', batch_size=args.batch_size, seq_len=args.seq_len)
    tokenizer = load_object('data/processed/ptb/tokenizer.pkl')

    model = LSTM(input_size, hidden_size, vocab_size=tokenizer)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for i_epoch in range(1, args.epochs + 1):
        for source, target in train_dataloader:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=32)

    args = parser.parse_args()
    main(args)
