import math
import torch
import torch.nn as nn


class LSTMPT(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,
                 embedding: nn.Embedding = None, decoder: nn.Module = None, num_layers: int = 1, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define embedding and decoder layers
        self.embedding = nn.Embedding(self.vocab_size, self.input_size) if embedding is None else embedding
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size) if decoder is None else decoder

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, inp, init_states=None):
        # import pdb; pdb.set_trace()

        if len(inp.shape) != 2:
            raise ValueError("Expected input tensor of order 2, but got order {} tensor instead".format(len(inp.shape)))

        x = self.embedding(inp)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        sequence_length, batch_size, _ = x.size()

        if init_states is None:
            hidden_seq, (h_t, c_t) = self.rnn(x)
        else:
            h_t, c_t = init_states
            hidden_seq, (h_t, c_t) = self.rnn(x, (h_t, c_t))

        output = self.decoder(hidden_seq.contiguous())

        if self.training:
            return output, (h_t, c_t)
        else:

            output_conf = torch.softmax(output, dim=-1)
            output_ids = torch.argmax(output_conf, dim=-1)  # [S, B]
            return output_ids, output_conf, (h_t, c_t)
