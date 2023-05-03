import copy
import os
import os.path as osp
import itertools

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt

from cprnn.utils import get_yaml_dict


def satisfies_conditions(root_a, root_b, catchall='any', verbose=False):

    for key, value in root_a.items():
        if isinstance(value, dict):
            if not satisfies_conditions(root_a[key], root_b[key]):
                return False
        else:
            if root_a[key] != root_b[key] and root_b[key] != catchall:
                if verbose:
                    print("  {} != {}".format(root_a[key], root_b[key]))
                return False

    return True


def get_leaf_value(root, addr_string):
    keys = addr_string.split('.')
    value = copy.deepcopy(root)
    while len(keys) > 0:
        value = value[keys.pop(0)]
    return value


def bpc_plot(args):

    experiments = dict()
    groups = dict()
    min_epochs = 25  # todo: change this quick fix
    # marker = itertools.cycle(('.', '^', 'o', '+', '*'))
    marker = itertools.cycle(('*', '.', 'o', '+', '.'))

    # Get Means/Stds
    for filename in sorted(os.listdir(args['visualization']['experiments'])):

        model_path = osp.join(args['visualization']['experiments'], filename, 'model_best.pth')
        if not osp.exists(model_path):
            print("  Error: {} | File Does Not Exist".format(model_path))
            continue

        try:
            dct_best = torch.load(
                osp.join(args['visualization']['experiments'], filename, 'model_best.pth'), map_location=torch.device('cpu')
            )
            dct_latest = torch.load(
                osp.join(args['visualization']['experiments'], filename, 'model_latest.pth'),
                map_location=torch.device('cpu')
            )
        except:
            print("  Error: {} | File Cannot Be Loaded".format(model_path))
            continue

        exp_cfg = get_yaml_dict(osp.join(args['visualization']['experiments'], filename, "configs.yaml"))

        # Configuring what to include in plot
        if not satisfies_conditions(exp_cfg, args, catchall='any'):
            # print("  Skipping: {} | Doesn't satisfy constraints".format(filename))
            continue

        elif dct_latest['epoch'] < args['visualization']['min_epochs']:
            print("  Error: {} | Too Few Epochs: {}".format(filename, dct_latest['epoch']))
            continue

        bpc = dct_best['test_metrics']['bpc']
        params = dct_best['num_params']
        epochs = dct_latest['epoch']

        print("Loaded {} | BPC: {:.2f} | Epochs: {}".format(filename, bpc, epochs))

        if filename.split("_tr")[0] not in experiments:
            experiments[filename.split("_tr")[0]] = (params, [bpc], exp_cfg, epochs)
        else:
            experiments[filename.split("_tr")[0]][1].append(bpc)

    for expname, (params, bpc_list, exp_cfg, epoch) in experiments.items():

        if len(bpc_list) < args['visualization']['min_trials']:
            print("  Error: {} | Too Few Trials: {}".format(expname, len(bpc_list)))
            continue

        bpc_list = bpc_list[:args['visualization']['min_trials']]
        bpc_mean = torch.tensor(bpc_list).mean().item()
        bpc_std = torch.tensor(bpc_list).std().item()
        hidden_size = exp_cfg['model']['hidden_size']

        print("Exp: {} | PTS: {} | BPC: {:.2f} +/- {:.2f} ({}) | Epochs: {}".format(
            expname, len(bpc_list), bpc_mean, bpc_std, [round(k, 2) for k in bpc_list], epoch
        ))

        group_name = ", ".join(["{}={}".format(attr_name.split('.')[-1][0], str(get_leaf_value(exp_cfg, attr_name))) for
                                attr_name in args['visualization']['group_by']])

        if group_name not in groups:
            groups[group_name] = ([params], [bpc_mean], [bpc_std], [hidden_size], [expname])
        else:
            groups[group_name][0].append(params)
            groups[group_name][1].append(bpc_mean)
            groups[group_name][2].append(bpc_std)
            groups[group_name][3].append(hidden_size)
            groups[group_name][4].append(expname)

    for group_name, (params, bpc_mean, bpc_std, hidden_size, filename) in groups.items():
        ids = np.argsort(params)
        params, bpc_mean, bpc_std, hidden_size = np.array(params)[ids], np.array(bpc_mean)[ids], \
                                                 np.array(bpc_std)[ids], np.array(hidden_size)[ids]


        {"plot": plt.plot, "scatter": plt.scatter}[args['visualization']['plt_type']](
            params, bpc_mean, label=group_name, marker=next(marker)
        )
        plt.errorbar(params, bpc_mean, yerr=bpc_std, ls='none')

        print("Group: {} | Num. Pts {}".format(group_name, len(bpc_mean)))
        print("  BPC {}".format(bpc_mean[:5]))
        print("  BPC STD {}".format(bpc_std[:5]))
        print("  PARAMS {}".format(params[:5]))
        print("  HIDDEN SZ {}".format(hidden_size[:5]))
        print("  FILES {}".format(filename[:5]))

        # plt.plot(params, bpc, marker=next(marker))

    plt.xlabel('Number of parameters')
    plt.ylabel('BPC')
    plt.xscale("log")
    plt.legend()
    plt.savefig(args['visualization']['output'])


@hydra.main(version_base=None, config_path="./", config_name="bpc_vs_params")
def main(cfg: DictConfig) -> None:

    args = OmegaConf.to_container(cfg, resolve=True)
    bpc_plot(args)


if __name__ == "__main__":
    main()