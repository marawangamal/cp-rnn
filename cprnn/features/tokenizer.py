import torch
import string


class CharacterTokenizer:
    def __init__(self, additional_chars: str = ""):

        self.additional_chars = additional_chars
        # chars = string.ascii_letters + " .,;'-!.?<>&" + string.digits + additional_chars
        chars = string.printable

        # Character to index and index to character maps
        self.char_to_ix_dct = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char_dct = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self):
        return len(self.char_to_ix_dct)

    def char_to_ix(self, char: str = None):
        return self.char_to_ix_dct[char]

    def ix_to_char(self, char: str = None):
        return self.ix_to_char_dct[char]

    def tokenize(self, sentence: str = None):
        return torch.tensor(list(map(self.char_to_ix, sentence)), dtype=torch.int32)