import math
import torch
import torch.nn as nn


class CPLSTM(nn.Module):
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
        self.rank = rank

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

        for t in range(sequence_length):
            x_t = x[:, t, :]

            # CP contraction
            gates = torch.multiply(h_t @ self.a, x_t @ self.b)  # [nxd][dx4r]
            gates = torch.matmul(
                # [n,4r] => [n,4,1,r]
                gates.reshape(batch_size, 4, 1, self.rank),

                # [4r,d] => [1,4,r,d]
                self.ct.reshape(1, 4, self.rank, self.hidden_size)
            ).squeeze()  # final dimension [n,4,d]

            funcs = [torch.sigmoid, torch.sigmoid, torch.tanh, torch.sigmoid]
            f_t, i_t, g_t, o_t = [funcs[k](gates[:, k, :]) for k in range(4)]

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


if __name__ == '__main__':

    input_size, hidden_size, cp_rank = 128, 128, 32
    cplstm = CPLSTM(input_size, hidden_size, cp_rank)

    batch_size, sequence_length = 32, 32
    x = torch.randn(batch_size, sequence_length, input_size)
    out = cplstm(x)

    # output
    # out[0].shape: torch.Size([32, 32, 128])
