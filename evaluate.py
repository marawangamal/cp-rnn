import math
import time

import logging
import os.path as osp

import torch
import torch.nn as nn

from cprnn.utils import load_object, AverageMeter, get_yaml_dict, load_weights
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
    logging.info("Device: {}".format(device))

    # Model
    model = _models[args["model"]["name"].lower()](vocab_size=tokenizer.vocab_size, **args["model"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dct = torch.load(osp.join(output_path, 'model_best.pth'), map_location=torch.device('cpu'))
    load_weights(model, dct)

    print("Epochs {} | Train Loss {:5.2f} | Train PPL {:5.2f} | Train BPC {:5.2f} | "
          "Valid Loss {:5.2f} | Valid PPL {:5.2f} | Valid BPC {:5.2f}".format(
        dct['epoch'],
        dct['train_metrics']['loss'], dct['train_metrics']['ppl'], dct['train_metrics']['loss'] / math.log(2),
        dct['valid_metrics']['loss'], dct['valid_metrics']['ppl'], dct['valid_metrics']['loss'] / math.log(2),
    ))

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Parallelize the model
    if torch.cuda.device_count() > 1:
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with torch.no_grad():

        elapsed_start = time.time()

        # Quantitative evaluation
        valid_metrics = evaluate(model, valid_dataloader, criterion, device=device)
        test_metrics = evaluate(model, test_dataloader, criterion, device=device)

        dct['test_metrics'] = test_metrics
        torch.save(dct, osp.join(output_path, "model_best.pth"))  # Update best model saved metrics

        # Qualitative prediction
        valid_sent_output, valid_sent_target, valid_sent_source = evaluate_qualitative(
            model, valid_dataloader, tokenizer, device,
        )
        test_sent_output, test_sent_target, test_sent_source = evaluate_qualitative(
            model, test_dataloader, tokenizer, device,
        )

        logging.info('Elapsed Time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.2f} '
                     '| test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.2f}'.format(
            (time.time() - elapsed_start),
            valid_metrics['loss'], valid_metrics['ppl'], valid_metrics['loss'] / math.log(2),
            test_metrics['loss'], test_metrics['ppl'], test_metrics['loss'] / math.log(2)
        ))

        valid_sent_output = valid_sent_output.transpose(1, 0)
        valid_sent_target = valid_sent_target.transpose(1, 0)
        valid_sent_source = valid_sent_source.transpose(1, 0)
        test_sent_output = test_sent_output.transpose(1, 0)
        test_sent_target = test_sent_target.transpose(1, 0)
        test_sent_source = test_sent_source.transpose(1, 0)

        valid_qaul_str = "Source:  \n{}  \nTarget:  \n{}  \nPrediction:  \n{}".format(
            "".join(valid_sent_source[:, 0]), "".join(valid_sent_target[:, 0]), "".join(valid_sent_output[:, 0])
        )

        test_qaul_str = "Source:  \n{}  \nTarget:  \n{}  \nPrediction:  \n{}".format(
            "".join(test_sent_source[:, 0]), "".join(test_sent_target[:, 0]), "".join(test_sent_output[:, 0])
        )

        sample_str = sample(model, size=100, prime='The', top_k=5, device=device, tokenizer=tokenizer)

        # Sample
        logging.info("Validation:\n{}".format(valid_qaul_str))
        logging.info("Test:\n{}".format(test_qaul_str))
        logging.info("Sample:\n{}".format(sample_str))


def sample(model, tokenizer, device=torch.device('cpu'), size=100, prime='The', top_k=5):
    # First off, run through the prime characters
    chars = [ch for ch in prime]

    model_alias = model.module if isinstance(model, nn.DataParallel) else model
    init_states = model_alias.init_hidden(batch_size=1, device=device)

    for ch in prime:
        inp = torch.tensor(tokenizer.char_to_ix(ch)).reshape(1, 1).to(device)
        output_id, init_states = model_alias.predict(inp, init_states, top_k=top_k, device=device)

    chars.append(tokenizer.ix_to_char(output_id.item()))

    # Now pass in the previous character and get a new one
    for ii in range(size):
        inp = torch.tensor(tokenizer.char_to_ix(chars[-1])).reshape(1, 1).to(device)
        output_id, init_states = model_alias.predict(inp, init_states, top_k=top_k, device=device)
        chars.append(tokenizer.ix_to_char(output_id.item()))

    return ''.join(chars)


def evaluate_qualitative(model, eval_dataloader, tokenizer: CharacterTokenizer, device: torch.device):
    with torch.no_grad():
        source, target = next(iter(eval_dataloader))
        source, target = source.to(device), target.to(device)
        output, _, _ = model(source)  # [bsz, seq, d_vocab]
        output = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        sentences_output = tokenizer.ix_to_char(output.cpu().detach().numpy())
        sentences_target = tokenizer.ix_to_char(target.cpu().detach().numpy())
        sentences_source = tokenizer.ix_to_char(source.cpu().detach().numpy())

    return sentences_output, sentences_target, sentences_source


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
