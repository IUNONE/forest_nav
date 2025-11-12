import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(MLP,self).__init__()

        self.num_layers = num_hidden_layers + 1
        h = [hidden_dim] * num_hidden_layers
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in  zip([input_dim] + h, h + [output_dim]))
    def forward(self, x)->torch.Tensor:
        for i, layer in  enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1  else layer(x)
        return x
    