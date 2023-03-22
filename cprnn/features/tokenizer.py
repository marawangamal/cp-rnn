from collections.abc import Iterable
import torch


class CharacterTokenizer:
    def __init__(self, tokens: Iterable[str] = None):
        """

        Args:
            tokens: (iterable) previously extracted tokens
        """
        # Character to index and index to character maps
        self._tokens = list(tokens) if not isinstance(tokens, list) else tokens
        self.char_to_ix_dct = {ch: i for i, ch in enumerate(tokens)}
        self.ix_to_char_dct = {i: ch for i, ch in enumerate(tokens)}

    @property
    def vocab_size(self):
        return len(self.char_to_ix_dct)

    @property
    def tokens(self):
        return self._tokens

    def char_to_ix(self, char: str = None):
        return self.char_to_ix_dct[char]

    def ix_to_char(self, char: str = None):
        return self.ix_to_char_dct[char]

    def tokenize(self, sentence: str = None):
        return torch.tensor(list(map(self.char_to_ix, sentence)), dtype=torch.int32)
