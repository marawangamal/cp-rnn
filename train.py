import math
import os
import time
import yaml

import logging
import os.path as osp
import argparse as argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from cprnn.utils import load_object, AverageMeter
from cprnn.models import LSTM, CPLSTM, CPRNN, SecondOrderRNNKR, LSTMPT
from cprnn.features.ptb_dataloader import PTBDataloader
from cprnn.features.tokenizer import CharacterTokenizer

_output_paths = {
    "models": "models"
}

_models = {
    "lstm": LSTM,
    "cplstm": CPLSTM,
    "cprnn": CPRNN,
    "2rnn": SecondOrderRNNKR,
    "lstmpt": LSTMPT
}


def main():
    stream = open("configs.yaml", 'r')
    args = yaml.safe_load(stream)
    for key, value in args.items():
        print(key + " : " + str(value))

    # Output location
    folder_name = "{}_e{}_l{}_b{}_s{}_r{}".format(
        args["model"]["name"], args["train"]["epochs"], str(args["train"]["lr"]).split('.')[-1],
        args["train"]["batch_size"], args["train"]["seq_len"], args["model"]["rank"]
    )

    output_path = osp.join("runs", osp.split(args["data"]["path"])[-1], folder_name)
    if not osp.exists(output_path):
        os.makedirs(output_path)

    with open(osp.join(output_path, 'configs.yaml'), 'w') as outfile:
        yaml.dump(args, outfile)

    writer = SummaryWriter(log_dir=output_path)

    # Logging configuration
    logging.basicConfig(filename=output_path,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # Data
    train_dataloader = PTBDataloader(
        osp.join(args["data"]["path"], 'train.pth'), batch_size=args["train"]["batch_size"],
        seq_len=args["train"]["seq_len"], batch_first=args["model"]["batch_first"]
    )
    valid_dataloader = PTBDataloader(
        osp.join(args["data"]["path"], 'valid.pth'), batch_size=args["train"]["batch_size"],
        seq_len=args["train"]["seq_len"], batch_first=args["model"]["batch_first"]
    )
    tokenizer = CharacterTokenizer(tokens=load_object('data/processed/ptb/tokenizer.pkl'))

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Device: {}".format(device))

    # Model
    model = _models[args["model"]["name"].lower()](vocab_size=tokenizer.vocab_size, **args["model"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["train"]["lr"])

    # Training
    best_valid_loss = None
    for i_epoch in range(1, args["train"]["epochs"] + 1):
        epoch_start_time = time.time()
        train_metrics = train(
            model, train_dataloader, optimizer, criterion, clip=args["train"]["grad_clip"], device=device,
        )
        valid_metrics = evaluate(model, valid_dataloader, criterion, device=device)

        logging.info('Epoch {:4d}/{:4d} | time: {:5.2f}s | train loss {:5.2f} | '
                     'train ppl {:8.2f} | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            i_epoch, args["train"]["epochs"], (time.time() - epoch_start_time), train_metrics['loss'],
            train_metrics['ppl'], valid_metrics['loss'], valid_metrics['ppl']
        ))

        # Get the gradients and log the histogram
        for name, param in model.named_parameters():
            writer.add_histogram(f"{name}.grad", param.grad, i_epoch)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_valid_loss or valid_metrics['loss'] < best_valid_loss:
            torch.save({
                'epoch': i_epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'torchrandom_state': torch.get_rng_state(),
                'train_metrics': valid_metrics,
                'valid_metrics': valid_metrics,
                'num_params': num_params,
                'config': args
            }, osp.join(output_path, "model_best.pth"))

            best_valid_loss = valid_metrics['loss']

        # Qualitative prediction
        train_sent_output, train_sent_target, train_sent_source = evaluate_qualitative(
            model, train_dataloader, tokenizer, device,
        )
        valid_sent_output, valid_sent_target, valid_sent_source = evaluate_qualitative(
            model, valid_dataloader, tokenizer, device,
        )

        if args["model"]["batch_first"]:
            valid_sent_output = valid_sent_output.transpose(1, 0)
            valid_sent_target = valid_sent_target.transpose(1, 0)
            valid_sent_source = valid_sent_source.transpose(1, 0)
            train_sent_output = train_sent_output.transpose(1, 0)
            train_sent_target = train_sent_target.transpose(1, 0)
            train_sent_source = train_sent_source.transpose(1, 0)

        valid_qaul_str = "Source:  \n{}  \nTarget:  \n{}  \nPrediction:  \n{}".format(
            "".join(valid_sent_source[:, 0]), "".join(valid_sent_target[:, 0]), "".join(valid_sent_output[:, 0])
        )

        train_qaul_str = "Source:  \n{}  \nTarget:  \n{}  \nPrediction:  \n{}".format(
            "".join(train_sent_source[:, 0]), "".join(train_sent_target[:, 0]), "".join(train_sent_output[:, 0])
        )

        sample_str = sample(model, size=100, prime='The', top_k=5, device=device, tokenizer=tokenizer)

        # Sample
        logging.info("Train:\n{}".format(train_qaul_str))
        logging.info("Validation:\n{}".format(valid_qaul_str))
        logging.info("Sample:\n{}".format(sample_str))

        # Logging
        for m in train_metrics.keys():
            writer.add_scalar("train/{}".format(m), train_metrics[m], i_epoch)
            writer.add_scalar("valid/{}".format(m), valid_metrics[m], i_epoch)

        writer.add_scalar("LR", args["train"]["lr"], i_epoch)
        writer.add_text('Valid', valid_qaul_str, i_epoch)
        writer.add_text('Train', train_qaul_str, i_epoch)
        writer.add_text('Sample', sample_str, i_epoch)

    writer.flush()
    writer.close()


def sample(model, tokenizer, device=torch.device('cpu'), size=100, prime='The', top_k=5):
    # First off, run through the prime characters
    chars = [ch for ch in prime]

    init_states = model.init_hidden(batch_size=1, device=device)

    for ch in prime:
        inp = torch.tensor(tokenizer.char_to_ix(ch)).reshape(1, 1).to(device)
        output_id, init_states = model.predict(inp, init_states, top_k=top_k, device=device)

    chars.append(tokenizer.ix_to_char(output_id.item()))

    # Now pass in the previous character and get a new one
    for ii in range(size):
        inp = torch.tensor(tokenizer.char_to_ix(chars[-1])).reshape(1, 1).to(device)
        output_id, init_states = model.predict(inp, init_states, top_k=top_k, device=device)
        chars.append(tokenizer.ix_to_char(output_id.item()))

    return ''.join(chars)


def evaluate_qualitative(model, eval_dataloader, tokenizer: CharacterTokenizer, device: torch.device):
    with torch.no_grad():
        source, target = next(iter(eval_dataloader))
        source, target = source.to(device), target.to(device)
        output, _ = model(source)  # [bsz, seq, d_vocab]
        output = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        sentences_output = tokenizer.ix_to_char(output.cpu().detach().numpy())
        sentences_target = tokenizer.ix_to_char(target.cpu().detach().numpy())
        sentences_source = tokenizer.ix_to_char(source.cpu().detach().numpy())

    return sentences_output, sentences_target, sentences_source


def evaluate(model, eval_dataloader, criterion, device):
    with torch.no_grad():
        model.to(device)
        loss_average_meter = AverageMeter()
        ppl_average_meter = AverageMeter()
        for inputs, targets in eval_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            output, _ = model(inputs)
            n_seqs_curr, n_steps_curr = output.shape[0], output.shape[1]
            loss = criterion(output.reshape(n_seqs_curr * n_steps_curr, -1),
                             targets.reshape(n_seqs_curr * n_steps_curr))

            loss_average_meter.add(loss.item())
            ppl_average_meter.add(torch.exp(loss).item())

    return {"loss": loss_average_meter.value,
            "ppl": ppl_average_meter.value}


def train(model, train_dataloader, optimizer, criterion, clip=5, device=torch.device('cpu')):
    model.train()
    model.to(device)
    loss_average_meter = AverageMeter()
    ppl_average_meter = AverageMeter()

    # for x, y in get_batches(data, n_seqs, n_steps):
    for i_batch, (inputs, targets) in enumerate(train_dataloader):  # [L, BS]

        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        output, _ = model.forward(inputs)

        n_seqs_curr, n_steps_curr = output.shape[0], output.shape[1]
        loss = criterion(output.reshape(n_seqs_curr * n_steps_curr, -1),
                         targets.reshape(n_seqs_curr * n_steps_curr))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        loss_average_meter.add(loss.item())
        ppl_average_meter.add(torch.exp(loss).item())

    return {"loss": loss_average_meter.value,
            "ppl": ppl_average_meter.value}


if __name__ == '__main__':
    main()

    """
    Commands
    
    // Semantic: toy rnn dataset generated with certain input size, hidden size vocab size rank
    python train.py -d data/processed/toy-2rnnkr-i32-h32-v16-r32
    
    python train.py -d data/processed/anna
    
    """
