import math

import torch
import torch.nn as nn


class SecondOrderRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden features.
        rank: Rank of cp factorization
    """
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, rank: int = 8,
                 embedding: nn.Embedding = None, decoder: nn.Module = None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank

        # Define embedding and decoder layers
        self.embedding = nn.Embedding(self.vocab_size, self.input_size) if embedding is None else embedding
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size) if decoder is None else decoder

        # Encoder using CP factors
        self.w = nn.Parameter(torch.Tensor(self.hidden_size + 1, self.input_size + 1, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inp: torch.LongTensor,
                init_state: torch.Tensor = None):

        if len(inp.shape) != 2:
            raise ValueError("Expected input tensor of order 2, but got order {} tensor instead".format(len(inp.shape)))

        x = self.embedding(inp)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        batch_size, sequence_length, _ = x.size()
        hidden_seq = []

        if init_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        else:
            h_t = init_state

        for t in range(sequence_length):
            x_t = x[t, :, :]

            h_prime = torch.cat((h_t, torch.ones(batch_size, 1)), dim=1)  # [B, D_h][D_h, R] => [B, R]
            x_prime = torch.cat((x_t, torch.ones(batch_size, 1)), dim=1)   # [B, D_i'][D_i', R] => [B, R]
            h_tnew = torch.sigmoid(torch.einsum("bi,bj,ijk->bk", h_prime, x_prime, self.w))
            hidden_seq.append(h_tnew.unsqueeze(0))
            h_t = h_tnew

        hidden_seq = torch.cat(hidden_seq, dim=0)
        output = self.decoder(hidden_seq.contiguous())

        if self.training:
            return output, h_t
        else:

            output_conf = torch.softmax(output, dim=-1)
            output_ids = torch.argmax(output_conf, dim=-1)  # [S, B]
            return output_ids, output_conf, h_t