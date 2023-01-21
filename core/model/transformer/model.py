import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from core.model.encoderlayer import MLP2Layer_Osmosis
from core.model.utils import _gen_positional_encoding

class Model(nn.Module):
    def __init__(self, encoder_layer, num_layers:int, hidden_size:int, threshold:float = 0.9, max_seq_len:int=10000, norm = None):
    
        super().__init__()
        
        timing_signal = torch.empty(1, num_layers + 1, hidden_size)
        nn.init.normal_(timing_signal,mean = 0.0, std = 0.1)
        self.timing_signal = nn.Parameter(timing_signal)

        self.max_hop = num_layers
        self.enc = encoder_layer
        self.norm = norm
        
        self.p = nn.Sigmoid()
        print("p", self.p)
        self.threshold = threshold

    def forward(self, state: Tensor, mem: Tensor, mask=None, src_key_padding_mask=None,  info = None):
        B, L, N = state.shape
        B, M, N = mem.shape
        
        halting_probability = torch.zeros((B,L), device = state.device)
        remainders = torch.zeros((B,L), device = state.device)
        n_updates = torch.zeros((B,L), device = state.device)
        
        previous_state = torch.zeros_like(state)
        condition = torch.tensor(True, dtype = torch.bool, device = "cpu")
        
        step = 0
        while(condition and step < self.max_hop):
            p = self.p(state[:,:,-1]).squeeze(-1)
            
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
            condition = condition.to("cpu")
                
            if info is not None:
                with info(name_scope = f"layer_{step}"):
                    state, mem = self.enc(state, mem, depth= step,src_mask=mask, src_key_padding_mask=src_key_padding_mask, info = info)
            else:
                state, mem = self.enc(state, mem, depth= step,src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                    
            # update running part in the weighted state and keep the rest
            # (1-update_weights) term gone To follow the correct implementation of Graves ACT paper.
            previous_state = ((state * update_weights.unsqueeze(-1)) + previous_state )
            step+=1
        
        x = previous_state
        if self.norm is not None:
            x = self.norm(x)
        return x
