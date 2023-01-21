import torch
import torch.nn as nn

class MLP2Layer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(normalized_shape = hidden_dim, eps=1e-8)
        self.act = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_dim, out_dim)
        self.prob = nn.Sigmoid()
        self.osmosis = nn.Dropout(p = 0.5)
        
    def forward(self, input, src_mask=None, src_key_padding_mask=None, info = None):
        x = self.fc_1(input)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc_2(x)
        if info is not None: info.store(act_prob_b4sig = x)
        x = self.prob(x)
        if info is not None: info.store(act_prob_b4osmosis = x)
        #x = nn.functional.dropout(x, p=0.5, training=True)
        x = self.osmosis(x)
        return x
