import math

import torch
import torch.nn as nn


class CPRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden features.
        rank: Rank of cp factorization
    """
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, rank: int = 8,
                 embedding: nn.Embedding = None, decoder: nn.Module = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank

        # Define embedding and decoder layers
        self.embedding = nn.Embedding(self.vocab_size, self.input_size) if embedding is None else embedding
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size) if decoder is None else decoder

        # Encoder using CP factors
        self.a = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.b = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.c = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
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

            a_prime = h_t @ self.a  # [B, D_h][D_h, R] => [B, R]
            b_prime = x_t @ self.b  # [B, D_i][D_i, R] => [B, R]
            h_t = torch.tanh(torch.einsum("br,br,hr->bh", a_prime, b_prime, self.c))
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        output = self.decoder(hidden_seq.contiguous())
        
        if self.train:
            return output, h_t
        else:
            output_conf = torch.softmax(output, dim=-1)
            output_ids = torch.argmax(output_conf, dim=-1)  # [S, B]
            return output_ids, output_conf
