import torch
from .tokenizer import CharacterTokenizer


class PTBDataloader:
    def __init__(self, dataset_path: str, batch_size: int = 32, seq_len: int = 32):
        # Since dataset is small, we will combine it all in a single vector
        self.dataset_ids = torch.load(dataset_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = self.dataset_ids.size(0) // self.batch_size

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.dataset_ids = torch.narrow(self.dataset_ids, 0, 0, self.num_batches * self.batch_size)

        # Split evenly into `batch_size` chunks [total//batch_size, batch_size]
        self.dataset_ids = self.dataset_ids.view(self.batch_size, self.num_batches).t().contiguous()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.num_batches:
            source = self.dataset_ids[self.n: self.n + self.seq_len]  # [L, BS]
            target = self.dataset_ids[self.n + 1: self.n + 1 + self.seq_len]  # [L, BS]
            self.n += self.seq_len + 1
            return source, target
        else:
            raise StopIteration


class PTBDataloaderOnline:
    def __init__(self, dataset: torch.utils.data.Dataset = None, batch_size: int = 32, seq_len: int = 32):
        # Since dataset is small, we will combine it all in a single vector
        self.dataset_ids = torch.zeros(sum([len(line) for line in dataset]), dtype=torch.int32)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = CharacterTokenizer()

        idx = 0
        for line in dataset:
            for char in line:
                self.dataset_ids[idx] = self.tokenizer.tokenize(char)
                idx += 1

        self.num_batches = self.dataset_ids.size(0) // self.batch_size

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.dataset_ids = torch.narrow(self.dataset_ids, 0, 0, self.num_batches * self.batch_size)

        # Split evenly into `batch_size` chunks [total//batch_size, batch_size]
        self.dataset_ids = self.dataset_ids.view(self.batch_size, self.num_batches).t().contiguous()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.num_batches:
            source = self.dataset_ids[self.n: self.n + self.seq_len]  # [L, BS]
            target = self.dataset_ids[self.n + 1: self.n + 1 + self.seq_len]  # [L, BS]
            self.n += self.seq_len + 1
            return source, target
        else:
            raise StopIteration
