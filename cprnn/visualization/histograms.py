import math
import time

import logging
import os.path as osp

import torch
import torch.nn as nn

from cprnn.utils import load_object, AverageMeter, get_yaml_dict
from cprnn.models import CPRNN, SecondOrderRNN, LSTM, MRNN, MIRNN
from cprnn.features.ptb_dataloader import PTBDataloader
from cprnn.features.tokenizer import CharacterTokenizer

_output_paths = {
    "models": "models"
}

_models = {
    "cprnn": CPRNN,
    "2rnn": SecondOrderRNN,
    "lstmpt": LSTM,
    "mrnn": MRNN,
    "mirnn": MIRNN
}


def load_weights(model, dct):
    if all(['module' in k for k in dct['model_state_dict'].keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in dct['model_state_dict'].items()}
    else:
        state_dict = dct['model_state_dict']
    model.load_state_dict(state_dict)


def main():

    # args for running eval
    eval_args = get_yaml_dict("configs.yaml")
    for key, value in eval_args.items():
        print(key + " : " + str(value))
    output_path = eval_args["eval"]["path"]

    # args of previously run experiment
    args = get_yaml_dict(osp.join(output_path, 'configs.yaml'))

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(osp.join(output_path, "logging-eval.txt")),
            logging.StreamHandler()
        ]
    )

    # Data
    valid_dataloader = PTBDataloader(
        osp.join(eval_args["data"]["path"], 'valid.pth'), batch_size=args["train"]["batch_size"],
        seq_len=args["train"]["seq_len"]
    )

    test_dataloader = PTBDataloader(
        osp.join(eval_args["data"]["path"], 'test.pth'), batch_size=args["train"]["batch_size"],
        seq_len=args["train"]["seq_len"]
    )

    tokenizer = CharacterTokenizer(tokens=load_object(osp.join(eval_args['data']['path'], 'tokenizer.pkl')))

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Model
    model = _models[args["model"]["name"].lower()](vocab_size=tokenizer.vocab_size, **args["model"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dct = torch.load(osp.join(output_path, 'model_best.pth'), map_location=torch.device('cpu'))
    load_weights(model, dct)

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Parallelize the model
    if torch.cuda.device_count() > 1:
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with torch.no_grad():

        # Quantitative evaluation
        valid_metrics = evaluate(model, valid_dataloader, criterion, device=device)
        test_metrics = evaluate(model, test_dataloader, criterion, device=device)


def evaluate(model, eval_dataloader, criterion, device):
    with torch.no_grad():
        loss_average_meter = AverageMeter()
        ppl_average_meter = AverageMeter()
        for inputs, targets in eval_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            output, _, hidden_seq = model(inputs)
            n_seqs_curr, n_steps_curr = output.shape[0], output.shape[1]
            loss = criterion(output.reshape(n_seqs_curr * n_steps_curr, -1),
                             targets.reshape(n_seqs_curr * n_steps_curr))

            if edges is None:
                hist, edges = torch.histogram(hidden_seq, bins=100)
            else:
                hist_new, edges = torch.histogram(hidden_seq, edges)
                hist += hist_new

            loss_average_meter.add(loss.item())
            ppl_average_meter.add(torch.exp(loss).item())

    return {"loss": loss_average_meter.value,
            "ppl": ppl_average_meter.value,
            "bpc": loss_average_meter.value / math.log(2),
            "hist": hist}



if __name__ == '__main__':
    main()

    """
    Commands

    // Semantic: toy rnn dataset generated with certain input size, hidden size vocab size rank
    python train.py -d data/processed/toy-2rnnkr-i32-h32-v16-r32

    python train.py -d data/processed/anna

    """
