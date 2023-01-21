import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from core.model.utils import _gen_positional_encoding

class Model(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers:int, hidden_size:int, max_seq_len:int=10000, norm = None):
    
        super().__init__()
        
        self.register_buffer('timing_signal',_gen_positional_encoding(hidden_size, max_len = (num_layers+1) )) # 1, max_len, num_layers 
        self.register_buffer('position_signal', _gen_positional_encoding(hidden_size, max_len = max_seq_len)) # 1, max_len, num_layers 
        #self.register_buffer('timing_signal', self.timing_signal)

        self.max_hop = num_layers
        self.enc = encoder_layer
        self.norm = norm

    def forward(self, state: Tensor,outer_step :int, mask=None, src_key_padding_mask=None):
        B, L, N = state.shape
        # mem: B, M, N
        
        position_signal = self.position_signal[:, :L, :]
        timing_signal = self.timing_signal
        
        step = 0
        while(step < self.max_hop):
            state = state + position_signal + self.timing_signal[:, step:step+1, :]
            state = self.enc(state,src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            step+=1
        
        x = state
        if self.norm is not None:
            x = self.norm(x)
        return x 