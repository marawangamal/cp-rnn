from typing import Union
import numpy as np


class CharacterTokenizer:
    """Facilitates Conversion between vocab <=> integers (tokenization)

    Args:
        tokens: (iterable) previously extracted tokens (i.e. list of characters or words)
    """
    def __init__(self, tokens: Union[list, tuple] = None):

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

    def ix_to_char(self, char: Union[str, np.ndarray] = None):

        if isinstance(char, np.ndarray):
            def f(x):
                return self.ix_to_char_dct[x]
            vf = np.vectorize(f)
            return vf(char)

        elif isinstance(char, str):
            return self.ix_to_char_dct[char]
        else:
            raise ValueError("Tokenizer expected either str or array as input but got {}".format(type(char)))

    def tokenize(self, sentence: str = None):
        return np.array(list(map(self.char_to_ix, sentence)), dtype=np.int32)
