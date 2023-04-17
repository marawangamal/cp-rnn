import math
import time
import os.path as osp
import argparse as argparse

import torch
import torch.nn as nn

from cprnn.utils import load_object, saveckpt
from cprnn.models import LSTM, CPLSTM
from cprnn.features.ptb_dataloader import PTBDataloader
from cprnn.features.tokenizer import CharacterTokenizer


_output_paths = {
    "data": "data/processed/ptb/"
}


def main(args):
    input_size = 128
    hidden_size = 256
    grad_clip = 0.25
    interval = 200

    # Data
    train_dataloader = PTBDataloader(
        osp.join(_output_paths['data'], 'train.pth'), batch_size=args.batch_size, seq_len=args.seq_len
    )
    valid_dataloader = PTBDataloader(
        osp.join(_output_paths['data'], 'valid.pth'), batch_size=args.batch_size, seq_len=args.seq_len
    )
    tokenizer = CharacterTokenizer(tokens=load_object('data/processed/ptb/tokenizer.pkl'))

    # Model
    model = CPLSTM(input_size, hidden_size, vocab_size=tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    best_val_loss = None
    for i_epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, optimizer, criterion, i_epoch, interval, grad_clip)
        val_loss = evaluate(model, valid_dataloader, criterion)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(
            i_epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
        ))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            saveckpt(model)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if args.opt == 'SGD' or args.opt == 'Momentum':
                args.lr /= 4.0
                for group in optimizer.param_groups:
                    group['lr'] = args.lr

        # Qualitative prediction
        sentences_output, sentences_target = evaluate_qualitative(model, valid_dataloader, tokenizer)
        print("\nQualitative:\n============\nTarget:\n{}\nPrediction:\n{}\n".format(
            "".join(sentences_target[:, 0]), "".join(sentences_output[:, 0])
        ))


def evaluate(model, eval_dataloader, criterion):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for i_batch, (source, target) in enumerate(eval_dataloader):
            output, _ = model(source)
            batch_size, seq_len, output_size = output.shape
            loss = criterion(output.view(seq_len * batch_size, -1), target.view(batch_size * seq_len))
            total_loss += loss.data

    return total_loss/(len(eval_dataloader) * eval_dataloader.batch_size)


def evaluate_qualitative(model, eval_dataloader, tokenizer: CharacterTokenizer):
    with torch.no_grad():
        model.eval()
        source, target = next(iter(eval_dataloader))
        output, _ = model(source)  # [seq, bsz, d_vocab]
        output = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        sentences_output = tokenizer.ix_to_char(output.cpu().detach().numpy())
        sentences_target = tokenizer.ix_to_char(target.cpu().detach().numpy())
    return sentences_output, sentences_target


def train(model, train_dataloader, optimizer, criterion, i_epoch, interval, grad_clip):
    """Loop over dataset"""
    model.train()
    iteration = 0
    total_loss = 0
    start_time = time.time()
    for i_batch, (source, target) in enumerate(train_dataloader):  # [L, BS]
        output, _ = model(source)
        batch_size, seq_len, output_size = output.shape

        loss = criterion(output.view(batch_size * seq_len, -1), target.view(seq_len * batch_size))
        optimizer.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.data

        if iteration % interval == 0 and i_batch > 0:
            cur_loss = total_loss / interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} '
                  '| loss {:5.5f} | ppl {:8.2f}'.format(i_epoch,
                                                        i_batch, len(train_dataloader),
                                                        args.lr,
                                                        elapsed * 1000 / interval,
                                                        cur_loss,
                                                        math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=32)

    args = parser.parse_args()
    main(args)
