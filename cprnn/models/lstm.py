import math
import torch
import torch.nn as nn


# Taken from: https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,
                 embedding: nn.Embedding = None, decoder: nn.Module = None, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define embedding and decoder layers
        self.embedding = nn.Embedding(self.vocab_size, self.input_size) if embedding is None else embedding
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size) if decoder is None else decoder

        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size) if decoder is None else decoder

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inp,
                init_states=None):
        """Assumes x is of shape (batch, sequence)"""

        if len(inp.shape) != 2:
            raise ValueError("Expected input tensor of order 2, but got order {} tensor instead".format(len(inp.shape)))

        x = self.embedding(inp)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        sequence_length, batch_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(sequence_length):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        output = self.decoder(hidden_seq.contiguous())

        if self.training:
            return output, (h_t, c_t)
        else:

            output_conf = torch.softmax(output, dim=-1)
            output_ids = torch.argmax(output_conf, dim=-1)  # [S, B]
            return output_ids, output_conf, (h_t, c_t)