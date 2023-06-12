from typing import Union

import numpy as np
import torch
import torch.nn as nn

from cprnn.features.tokenizer import CharacterTokenizer


class LSTMPT(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, use_embedding: bool = False,
                 num_layers: int = 2, tokenizer: CharacterTokenizer = None, batch_first: bool = True,
                 dropout: float = 0.5, **kwargs):
        super().__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define embedding and decoder layers

        if use_embedding:
            self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        else:
            # One hot version
            self.embedding = lambda x: torch.nn.functional.one_hot(x, vocab_size).float()
            self.input_size = vocab_size

        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.vocab_size)
        )

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           dropout=self.dropout)

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        (h, c) = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), \
                 torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h, c

    def predict(self, inp: Union[torch.LongTensor, str], init_states: tuple = None, top_k: int = 1,
                device=torch.device('cpu')):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer not defined. Please provide a tokenizer to the model.")
                x = torch.tensor(self.tokenizer.char_to_ix(inp)).reshape(1, 1).to(device)
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, inp: torch.LongTensor, init_states: tuple = None):

        if self.batch_first:
            inp = inp.transpose(0, 1)

        if len(inp.shape) != 2:
            raise ValueError("Expected input tensor of order 2, but got order {} tensor instead".format(len(inp.shape)))

        x = self.embedding(inp)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        sequence_length, batch_size, _ = x.size()

        if init_states is None:
            hidden_seq, (h_t, c_t) = self.rnn(x)
        else:
            h_t, c_t = init_states
            device = next(self.parameters()).device
            h_t, c_t = h_t.to(device), c_t.to(device)
            hidden_seq, (h_t, c_t) = self.rnn(x, (h_t, c_t))

        output = self.decoder(hidden_seq.contiguous())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_t, c_t)

