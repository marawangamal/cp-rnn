from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):

    def __init__(self, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001, tokenizer=None, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_input = tokenizer.vocab_size
        self.chars = tokenizer.char_to_ix_dct.keys()
        self.lr = lr

        ## Define the LSTM
        self.lstm = nn.LSTM(self.n_input, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.embedding = lambda x: torch.nn.functional.one_hot(x, self.n_input).float()

        ## Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## Define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

        # Initialize the weights
        # self.init_weights()

    def forward(self, x, hc):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hc`. '''

        if not self.batch_first:
            x = x.transpose(0, 1)

        x = self.embedding(x)  # [B, S] = > [B, S, Din]

        ## Get x, and the new hidden state (h, c) from the lstm
        x, (h, c) = self.lstm(x, hc)

        ## Ppass x through the dropout layer
        x = self.dropout(x)

        # Stack up LSTM outputs using view
        # x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)

        ## Put x through the fully-connected layer
        x = self.fc(x)

        # Return x and the hidden state (h, c)

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x, (h, c)

    def predict1(self, char, h=None, top_k=None, device=torch.device('cpu')):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.tokenizer.char_to_ix(char)]])
        inputs = torch.from_numpy(x).to(device)

        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data

        # import pdb; pdb.set_trace()
        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.cpu().numpy().squeeze()

        p = p.cpu().numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())

        return self.tokenizer.ix_to_char(char), h


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

            if np.prod(output.shape[:2]) != 1:
                raise ValueError("Higher order predictions not supported yet.")

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()

            # k_star = torch.randint(0, top_k, (1, 1)).item()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1

        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs, device=torch.device('cpu')):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_().to(device),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_().to(device))