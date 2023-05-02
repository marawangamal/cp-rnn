import pickle

import yaml


def save_object(obj, filename):
    with open(filename, 'wb') as out_file:  # Overwrites any existing file.
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp_file:  # Overwrites any existing file.
        out = pickle.load(inp_file)
    return out


def saveckpt(model, epoch, optimizer):
    pass


class AverageMeter:
    def __init__(self):
        self.sum = None
        self.pts = None
        self.first = True

    def add(self, val, n=1):
        if self.first:
            self.sum = val
            self.pts = n
            self.first = False
        else:
            self.sum += val
            self.pts += n

    @property
    def value(self):
        return self.sum / self.pts


def get_yaml_dict(yaml_path="configs.yaml"):
    with open(yaml_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_weights(model, dct):
    if all(['module' in k for k in dct['model_state_dict'].keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in dct['model_state_dict'].items()}
    else:
        state_dict = dct['model_state_dict']
    model.load_state_dict(state_dict)