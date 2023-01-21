import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.model.encoderlayer import MLP2Layer_Osmosis
import copy

def _gen_positional_encoding(d_model: int, max_len: int = 1000):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe # 1, max_len, d_model

class Model(nn.Module):
    def __init__(self, encoder_layer, num_layers:int, num_dcbs:int, hidden_size:int, threshold:float = 0.9, max_seq_len:int=10000, norm = None):
    
        super().__init__()
        
        self.register_buffer('timing_signal',_gen_positional_encoding(hidden_size, max_len = (num_layers+1) * num_dcbs )) # 1, max_len, num_layers 
        self.register_buffer('position_signal', _gen_positional_encoding(hidden_size, max_len = max_seq_len)) # 1, max_len, num_layers 
        #self.register_buffer('timing_signal', self.timing_signal)

        self.max_hop = num_layers
        self.enc = encoder_layer
        self.norm = norm
        
        self.p = MLP2Layer_Osmosis(
                in_dim = hidden_size, 
                hidden_dim = hidden_size, 
                out_dim = 1 
            )
        self.threshold = threshold

    def forward(self, state, outer_step, mask=None, src_key_padding_mask=None,  info = None):
        B, L, N = state.shape
        
        halting_probability = torch.zeros((B,L), device = state.device)
        remainders = torch.zeros((B,L), device = state.device)
        n_updates = torch.zeros((B,L), device = state.device)
        
        position_signal = self.position_signal[:, :L, :]
        timing_signal = self.timing_signal
        
        previous_state = torch.zeros_like(state)
        condition = torch.tensor(True, dtype = torch.bool, device = "cpu")
        encoder = self.enc
        
        step = 0
        while(condition and step < self.max_hop):
            timing_signal = self.timing_signal[:, self.max_hop * outer_step + step:self.max_hop * outer_step + step+1, :]
            
            #print(step, state.device, position_signal.device, timing_signal.device)
            state = state + position_signal
            state = state + timing_signal
            
            p = self.p(state).squeeze(-1)
                
            
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()
            
            pressure = halting_probability + p * still_running

            # Mask of inputs which halted at this step
            new_halted = ( pressure > self.threshold).float() * still_running
            
            # Mask of inputs which haven't halted, and didn't halt this step
            continue_running = ( pressure <= self.threshold).float() * still_running
            
            update_probs = ((1 - halting_probability) * new_halted + p * continue_running)
            
            halting_probability = halting_probability + update_probs
            
            update_weights = update_probs
            
            condition = torch.any(torch.lt(halting_probability,self.threshold))
                
            state = encoder(state, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                    
            # update running part in the weighted state and keep the rest
            # (1-update_weights) term gone To follow the correct implementation of Graves ACT paper.
            previous_state = ((state * update_weights.unsqueeze(-1)) + previous_state )
            step+=1
        
        x = previous_state
        if self.norm is not None:
            x = self.norm(x)
        return x 