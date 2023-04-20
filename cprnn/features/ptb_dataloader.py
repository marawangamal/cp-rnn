import torch


class PTBDataloader:
    def __init__(self, dataset_path: str, batch_size: int = 32, seq_len: int = 32):
        # Since dataset is small, we load it all into a single vector
        self.dataset_ids = torch.load(dataset_path)  # entire dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = self.dataset_ids.size(0) // self.batch_size

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.dataset_ids = torch.narrow(self.dataset_ids, 0, 0, self.num_batches * self.batch_size)

        # Split evenly into `batch_size` chunks [total//batch_size, batch_size]
        self.dataset_ids = self.dataset_ids.view(self.batch_size, self.num_batches).t().contiguous()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batches:

            remaining = len(self.dataset_ids[self.n:])
            source = self.dataset_ids[self.n: self.n + min(self.seq_len, remaining - 1)]  # [L, BS]
            target = self.dataset_ids[self.n + 1: self.n + min(1 + self.seq_len, remaining)]  # [L, BS]
            self.n += self.seq_len + 1
            return source, target
        else:
            raise StopIteration
