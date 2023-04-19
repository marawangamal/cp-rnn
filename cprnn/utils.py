import pickle


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
        self.sum = 0
        self.pts = 0

    def add(self, val):
        self.sum += val
        self.pts += 1

    @property
    def value(self):
        return self.sum / self.pts



