import math
import os
import time
import os.path as osp
import argparse as argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from cprnn.utils import load_object, AverageMeter
from cprnn.models import LSTM, CPLSTM, CPRNN
from cprnn.features.ptb_dataloader import PTBDataloader
from cprnn.features.tokenizer import CharacterTokenizer


# _data_paths = {
#     "ptb": "data/processed/ptb",
#     "toy8": "data/processed/toy-rnn8"  # rank 8
# }

_output_paths = {
    "models": "models"
}

_models = {
    "lstm": LSTM,
    "cplstm": CPLSTM,
    "cprnn": CPRNN
}


def main(args):
    input_size = 128
    hidden_size = 256
    grad_clip = 0.25
    interval = 200

    # Data
    train_dataloader = PTBDataloader(
        osp.join(args.dataset, 'train.pth'), batch_size=args.batch_size, seq_len=args.seq_len
    )
    valid_dataloader = PTBDataloader(
        osp.join(args.dataset, 'valid.pth'), batch_size=args.batch_size, seq_len=args.seq_len
    )
    tokenizer = CharacterTokenizer(tokens=load_object('data/processed/ptb/tokenizer.pkl'))

    # Model
    model = _models[args.m.lower()](input_size, hidden_size, vocab_size=tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Output location
    if not osp.exists(osp.join(_output_paths['models'], args.dataset)):
        os.makedirs(osp.join(_output_paths['models'], args.dataset))

    filename = "{}_e{}_l{}_b{}_s{}_d{}".format(args.m, args.e, str(args.l).split('.')[-1], args.b, args.s, args.d)
    output_path = osp.join("runs", args.dataset, filename)
    writer = SummaryWriter(log_dir=output_path)

    # Training
    best_valid_loss = None
    for i_epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_metrics = train(model, train_dataloader, optimizer, criterion, i_epoch, interval, grad_clip)
        valid_metrics = evaluate(model, valid_dataloader, criterion)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(
            i_epoch, (time.time() - epoch_start_time), valid_metrics['loss'], math.exp(valid_metrics['loss'])
        ))

        # Logging
        writer.add_scalar("Loss/train", train_metrics['loss'], i_epoch)
        writer.add_scalar("Loss/valid", valid_metrics['loss'], i_epoch)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_valid_loss or valid_metrics['loss'] < best_valid_loss:

            torch.save({
                'epoch': i_epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'torchrandom_state': torch.get_rng_state(),
                'train_metrics': valid_metrics,
                'valid_metrics': valid_metrics,
            }, output_path)

            best_valid_loss = valid_metrics['loss']

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

    writer.flush()
    writer.close()


def evaluate(model, eval_dataloader, criterion):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for i_batch, (source, target) in enumerate(eval_dataloader):
            output, _ = model(source)
            batch_size, seq_len, output_size = output.shape
            loss = criterion(output.view(seq_len * batch_size, -1), target.view(batch_size * seq_len))
            total_loss += loss.data

    eval_metrics = {
        "loss": total_loss / (len(eval_dataloader) * eval_dataloader.batch_size)
    }

    return eval_metrics


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
    train_loss_average_meter = AverageMeter()
    for i_batch, (source, target) in enumerate(train_dataloader):  # [L, BS]
        output, _ = model(source)
        batch_size, seq_len, output_size = output.shape

        loss = criterion(output.view(batch_size * seq_len, -1), target.view(seq_len * batch_size))
        optimizer.zero_grad()
        loss.backward()

        # Log
        train_loss_average_meter.add(loss.item())

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.data

        if iteration % interval == 0:
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

        train_metrics = {
            "loss": train_loss_average_meter.value
        }

        return train_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('-m', '--model', type=str, default='cprnn', choices=list(_models.keys()))
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-l', '--lr', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-s', '--seq_len', type=int, default=32)
    parser.add_argument('-d', '--dataset', type=str, default='data/processed/ptb',
                        help="path to folder containing train.pth, valid.pth")

    args = parser.parse_args()
    main(args)

    """
    Commands
    
    // Semantic: toy rnn dataset generated with certain input size, hidden size vocab size rank
    python train.py -d data/processed/toy-rnn-i8-h8-v4-r4  
    
    """
