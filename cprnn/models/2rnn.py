import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class SecondOrderRNN(nn.Module):
    """
    Implements a 2RNN : 
    h_t = phi(A (h^T \otimes x^T)^T + Ux + Vh + b)
    y_t = sigma(W h_t + c)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gate = nn.ReLU()
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.input_size = input_size                             
        self.output_size = output_size

        self.A = nn.Parameter(torch.empty((hidden_size, input_size*hidden_size)))
        self.U = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.V = nn.Parameter(torch.empty((hidden_size, hidden_size)))
        self.b = nn.Parameter(torch.empty(hidden_size))
        self.W = nn.Parameter(torch.empty((output_size, hidden_size)))
        self.c = nn.Parameter(torch.empty(output_size))  
        self.init_weights()
    
    def khatri_rao(a, b):
        """
        Khatri-Rao product is a broadcasted kronecker product
        """
        batch_input_size, input_size = a.shape
        batch_hidden_size, hidden_size = b.shape
        assert batch_input_size == batch_hidden_size, "Batch size of the input and hidden should be equal"
        res = a.unsqueeze(-1) * b.unsqueeze(-2)
        res = res.reshape(batch_input_size, input_size*hidden_size)
        return res

    def init_weights(self):
        k = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-k, k)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        h_0 = torch.normal(mean = 0, std = 0.00001, size=(batch_size, self.hidden_size))
        if torch.cuda.is_available(): h_0.cuda()
        return h_0

    def recurrent_layer(self, x_input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        x_input.shape  = (batch_size, input_size) 
        hidden/h.shape = (batch_size, hidden_size) 
        """
        khatri_rao_input = self.khatri_rao(x_input, hidden)
        h = self.gate(F.linear(khatri_rao_input, self.A, bias=None)
                        + F.linear(hidden, self.V, self.b)  
                        + F.linear(x_input.float(), self.U,bias=None)) 
        return h 

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """ 
        inputs.shape = (seq_len, batch_size, input_size) 
        hidden.shape = (batch_size, hidden_size) -> h_0 
        hidden_out.shape = (seq_len, batch_size, hidden_size) 
        y_out.shape = (seq_len, batch_size, out_size) 
        """
        hidden_seq = []
        y_seq = []
        for t, x in enumerate(inputs): 
            hidden = self.recurrent_layer(x, hidden)
            y = F.linear(hidden, self.W, self.c)
            hidden_seq.append(hidden)
            y_seq.append(y)
        hidden_out = torch.stack(hidden_seq)
        y_out = torch.stack(y_seq)
        return y_out, hidden_out 