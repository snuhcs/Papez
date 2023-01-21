import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

class Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        num_spks=2,
        **kwargs,
    ):  
        super().__init__()
        self.num_spks = num_spks

        self.out_channels = out_channels

        self.embed = nn.Sequential(
                          nn.Conv1d(
                              in_channels = out_channels+1, 
                              out_channels = out_channels,
                              kernel_size = 1,
                          ),
                          nn.InstanceNorm1d(out_channels, affine=True),
                          nn.PReLU(),
                          nn.Conv1d(
                              in_channels = out_channels, 
                              out_channels = out_channels,
                              kernel_size = 1,
                          ),
                        )
        self.dislodge = nn.Sequential(
                            nn.Conv1d(
                              in_channels = out_channels, 
                              out_channels = out_channels* self.num_spks,
                              kernel_size = 1,
                            ),
                            nn.InstanceNorm1d(out_channels* self.num_spks, affine=True),
                            nn.PReLU(),
                            nn.Conv1d(
                              in_channels = out_channels* self.num_spks, 
                              out_channels = out_channels* self.num_spks,
                              kernel_size = 1,
                            ),
                            nn.Tanh()
                        )
        self.transformer = intra_model


    def forward(self, x, mem, info = None):
        x_mem = self.embed(torch.cat((x,mem),dim = 2) ) # B, N, L+M
        x_mem = torch.permute(x_mem, (0,2,1)) # B, L+M, N
        
        x, mem = x_mem[:,:x.shape[-1],:], x_mem[:,-mem.shape[-1]:,:] # (B, L, N), (B, M, N)
        if info is not None: x = self.transformer(x, mem, info = info) # B, L, N
        else: x = self.transformer(x, mem) # B, L, N
        x = torch.permute(x, (0,2,1)) # B, N, L
        
        x = self.dislodge(x) # B, 3, L_
        
        B, _, L = x.shape
        return torch.reshape(x, (B, self.num_spks, self.out_channels, L)) # B, spks, N, L

class Model(nn.Module):
    def __init__(
        self,
        intra_model,
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_numspks=4,
        num_memory_slots = 10,
        **kwargs,
    ):        
        super().__init__()
        self.encoder = nn.Sequential(
                          nn.Conv1d(
                            in_channels = encoder_in_nchannels, 
                            kernel_size=encoder_kernel_size,
                            out_channels = encoder_out_nchannels,
                            stride=encoder_kernel_size // 2,
                            padding='valid',
                          ),
                          nn.InstanceNorm1d(encoder_out_nchannels, affine=True),
                          nn.LeakyReLU(),
                          nn.Conv1d(
                            in_channels = encoder_out_nchannels, 
                            out_channels = encoder_out_nchannels,
                            kernel_size = 1,
                          ),
                        )
        
        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            num_spks=masknet_numspks,
        )
        self.decoder = nn.Sequential(
                            nn.Conv1d(
                              in_channels = encoder_out_nchannels, 
                              out_channels = encoder_out_nchannels,
                              kernel_size = 1,
                            ),
                            nn.InstanceNorm1d(encoder_out_nchannels, affine=True),
                            nn.LeakyReLU(),
                            nn.ConvTranspose1d(
                                in_channels = encoder_out_nchannels,
                                out_channels=encoder_in_nchannels,
                                kernel_size=encoder_kernel_size,
                                stride=encoder_kernel_size // 2,
                                bias=True,
                            ),
                        )
        
        self.register_buffer('memory_slots',self._memory_slots(encoder_out_nchannels + 1, num_memory_slots)) # 1, N, M 

        self.num_spks = masknet_numspks
        
    def _memory_slots(self, N, M):
        mem = torch.unsqueeze(torch.eye(N, M), 0)
        mem[:,-1,:] = 1 # Memory Tag
        return mem

    def forward(self, mix, info = None):
        B, _ , T_origin = mix.shape # B, 1, L
        mix_w = self.encoder(mix) # B, N, L'
        mix_w_tag = F.pad(mix_w, (0,0,0,1), "constant", 0) # Sequence Tag  (B, N+1, L')
        memory_slots = self.memory_slots.repeat(B,1,1)# B, N+1, M
        
        if info is not None: est_mask = self.masknet(mix_w_tag,memory_slots, info = info) #  B, n_spks, N, L'
        else:est_mask = self.masknet(mix_w_tag,memory_slots) #  B, n_spks, N, L'
        
        mix_w = torch.tile(torch.unsqueeze(mix_w, dim = 1), (1,self.num_spks, 1, 1)) #  B, n_spks, N-1, L'
        sep_h = mix_w * est_mask  # B, n_spks, N-1, L'

        B, n_spks, N_1, L = sep_h.shape
        sep_h = torch.reshape(sep_h, ( B * n_spks, N_1, L)) # ( B, n_spks, N-1, L')
        logits = self.decoder(sep_h) # ( B, n_spks, 1, _L)
        if info is not None: info.store(logits = logits)
        estimate = torch.reshape(logits, ( B, n_spks, -1)) # (B, n_spks, _L)
        
        T_est = estimate.shape[2]
        if  T_origin > T_est:
            estimate = F.pad(estimate, (0, T_origin - T_est,  0, 0, 0, 0), "constant", 0)
        else:
            estimate = estimate[:, :, :T_origin]
            
        return estimate


