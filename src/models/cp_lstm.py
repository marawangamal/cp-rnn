import math
import torch
import torch.nn as nn


# Taken from: https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
class CustomLSTM(nn.Module):
    """CP-Factorized LSTM.

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden features.
        rank: Rank of cp factorization
    """
    def __init__(self, input_size: int, hidden_size: int, rank: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # cp factors
        self.a = nn.Parameter(torch.Tensor(hidden_size, rank * 4))
        self.b = nn.Parameter(torch.Tensor(input_size, rank * 4))
        self.ct = nn.Parameter(torch.Tensor(rank * 4, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, sequence_length, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        hs = self.hidden_size
        for t in range(sequence_length):
            x_t = x[:, t, :]

            # batch the computations into a single matrix multiplication
            gates = torch.multiply(h_t @ self.a, x_t @ self.b)
            funcs = [torch.sigmoid, torch.sigmoid, torch.tanh, torch.sigmoid]
            f_t, i_t, g_t, o_t = [funcs[k](gates[:, k, :]) for k in range(4)]

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
