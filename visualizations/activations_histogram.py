import os
from os import path as osp

import torch
from matplotlib import pyplot as plt

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cprnn.features.ptb_dataloader import PTBDataloader
from cprnn.features.tokenizer import CharacterTokenizer
from cprnn.utils import load_object, get_yaml_dict, load_weights
from cprnn.models import CPRNN, SecondOrderRNN, LSTM, MRNN, MIRNN


_models = {
    "cprnn": CPRNN,
    "2rnn": SecondOrderRNN,
    "lstmpt": LSTM,
    "mrnn": MRNN,
    "mirnn": MIRNN
}


def activation_histogram_plot(args, n_bins=100):

    # Data
    tokenizer = CharacterTokenizer(tokens=load_object(osp.join(args['data']['path'], 'tokenizer.pkl')))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if not osp.exists(args['visualization']['output']):
        os.makedirs(args['visualization']['output'])

    min_epochs = 25  # todo: change this quick fix
    for foldername in os.listdir(args['visualization']['experiments']):

        model_path = osp.join(args['visualization']['experiments'], foldername, 'model_best.pth')
        if not osp.exists(model_path):
            print("  Skipping {} | File Does Not Exist".format(model_path))
            continue

        dct = torch.load(
            osp.join(args['visualization']['experiments'], foldername, 'model_best.pth'), map_location=torch.device('cpu')
        )

        exp_cfg = get_yaml_dict(osp.join(args['visualization']['experiments'], foldername, "configs.yaml"))
        valid_dataloader = PTBDataloader(
            osp.join(args["data"]["path"], 'valid.pth'), batch_size=exp_cfg["train"]["batch_size"],
            seq_len=exp_cfg["train"]["seq_len"]
        )

        # Configuring what to include in plot
        if dct['epoch'] < min_epochs:
            print("  Skipping: {} | Too Few Epochs: {}".format(foldername, dct['epoch']))
            continue

        # Model
        model = _models[exp_cfg["model"]["name"].lower()](vocab_size=tokenizer.vocab_size, **exp_cfg["model"])
        model.to(device)
        load_weights(model, dct)

        with torch.no_grad():

            hist, bins = torch.zeros(100), None
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output, _, hidden_seq = model(inputs)

                # import pdb; pdb.set_trace()
                hist_new, bins = torch.histogram(hidden_seq.reshape(-1).detach().cpu(),
                                                 bins=n_bins if bins is None else bins)
                hist += hist_new

        # import pdb; pdb.set_trace()
        plt.bar(bins[:-1], hist, color=(0.2, 0.4, 0.6, 0.6))
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(osp.join(args['visualization']['output'], foldername + '.png'))

    return hist


@hydra.main(version_base=None, config_path="./", config_name="activations_histogram")
def main(cfg: DictConfig) -> None:

    args = OmegaConf.to_container(cfg, resolve=True)
    activation_histogram_plot(args)


if __name__ == "__main__":
    main()