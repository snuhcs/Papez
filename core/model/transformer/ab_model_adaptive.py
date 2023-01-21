import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _gen_positional_encoding(d_model: int, max_len: int = 1000):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe # 1, max_len, d_model

class Model(nn.Module):
    def __init__(self, encoder_layer,  num_layers:int, num_dcbs:int, hidden_size:int, threshold:float = 0.9, max_seq_len:int=10000, norm = None):
    
        super().__init__()
        
        self.register_buffer('timing_signal',_gen_positional_encoding(hidden_size, max_len = num_layers * num_dcbs)) # 1, max_len, num_layers 
        self.register_buffer('position_signal', _gen_positional_encoding(hidden_size, max_len = max_seq_len)) # 1, max_len, num_layers 
        #self.register_buffer('timing_signal', self.timing_signal)

        self.max_hop = num_layers
        self.enc = encoder_layer
        self.norm = norm
        
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = threshold

    def forward(self, state, outer_step, mask=None, src_key_padding_mask=None,  info = None):
        B, L, N = state.shape
        halting_probability = torch.zeros((B,L), device = state.device)
        remainders = torch.zeros((B,L), device = state.device)
        n_updates = torch.zeros((B,L), device = state.device)
        
        position_signal = self.position_signal[:, :L, :]
        timing_signal = self.timing_signal
        
        previous_state = torch.zeros_like(state)
        
        condition = torch.any(torch.logical_and(torch.lt(halting_probability,self.threshold), torch.lt(n_updates , self.max_hop))).cpu()
        step = 0
        while(condition):
            # Add timing signal
            state = state + position_signal
            state = state + self.timing_signal[:, self.max_hop * outer_step + step:self.max_hop * outer_step + step+1, :]
            #.unsqueeze(1)#.repeat(1,inputs.shape[1],1).type_as(inputs.data)
            
            p_bef = self.p(state)
            p = self.sigma(p_bef).squeeze(-1)
            if info is not None: 
                info.store(**{f"act_prob_b4sig_step{step}" : p_bef})
                info.store(**{f"act_prob_step{step}" : p})
                print(info.path, " steps:", step)
            
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted
            
            condition = torch.any(torch.logical_and(torch.lt(halting_probability,self.threshold), torch.lt(n_updates , self.max_hop))).cpu()
            

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            '''
            if info is not None:
                with info(name_scope = f"layer_{step}"):
                    state = self.enc(state, src_mask=mask, src_key_padding_mask=src_key_padding_mask, info = info)
                    info.store(halting_probability = halting_probability)
                    info.store(state = info.unfold(torch.permute(torch.unsqueeze(state, 0), (0, 3, 2, 1) )))
            else:
            '''
                
            state = self.enc(state, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                    

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        
        x = previous_state
        if self.norm is not None:
            x = self.norm(x)
        return x #, (remainders, n_updates)