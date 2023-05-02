import copy
import os
import os.path as osp

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt

from cprnn.features.ptb_dataloader import PTBDataloader
from cprnn.features.tokenizer import CharacterTokenizer
from cprnn.utils import get_yaml_dict, AverageMeter
from cprnn.models import CPRNN, SecondOrderRNN, LSTM, MRNN, MIRNN


_models = {
    "cprnn": CPRNN,
    "2rnn": SecondOrderRNN,
    "lstmpt": LSTM,
    "mrnn": MRNN,
    "mirnn": MIRNN
}


def satisfies_conditions(root_a, root_b, catchall='any'):

    for key, value in root_a.items():
        if isinstance(value, dict):
            if not satisfies_conditions(root_a[key], root_b[key]):
                return False
        else:
            if root_a[key] != root_b[key] and root_b[key] != catchall:
                return False

    return True


def get_leaf_value(root, addr_string):
    keys = addr_string.split('.')
    value = copy.deepcopy(root)
    while len(keys) > 0:
        value = value[keys.pop(0)]
    return value


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


def get_hists():

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


@hydra.main(version_base=None, config_path="./", config_name="visualization_configs")
def main(cfg: DictConfig) -> None:

    args = OmegaConf.to_container(cfg, resolve=True)
    for key, value in args.items():
        print(key + " : " + str(value))

    groups = dict()
    min_epochs = 25  # todo: change this quick fix
    for filename in os.listdir(args['visualization']['root']):

        dct = torch.load(
            osp.join(args['visualization']['root'], filename, 'model_best.pth'), map_location=torch.device('cpu')
        )
        exp_cfg = get_yaml_dict(osp.join(args['visualization']['root'], filename, "configs.yaml"))

        # Configuring what to include in plot
        if not satisfies_conditions(exp_cfg, args, catchall='any') or dct['epoch'] < min_epochs:
            print("  Skipping: {} | Epochs: {}".format(filename, dct['epoch']))
            continue

        bpc = dct['test_metrics']['bpc']
        params = dct['num_params']

        print("Exp: {} | Epochs: {}".format(filename, dct['epoch']))

        group_name = ", ".join([str(get_leaf_value(exp_cfg, attr_name)) for attr_name in args['visualization']['group_by']])

        if group_name not in groups:
            groups[group_name] = ([params], [bpc])
        else:
            groups[group_name][0].append(params)
            groups[group_name][1].append(bpc)

    for group_name, (params, bpc) in groups.items():
        plt.scatter(params, bpc, label=group_name)

    plt.xlabel('Number of parameters')
    plt.ylabel('BPC')
    plt.legend()
    plt.savefig(args['visualization']['output_filename'])


if __name__ == "__main__":
    main()