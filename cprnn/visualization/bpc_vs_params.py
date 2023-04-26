import copy
import os
import os.path as osp

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt

from cprnn.utils import get_yaml_dict


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