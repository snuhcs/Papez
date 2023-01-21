import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
    
#https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dims: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dims, 2) * (-math.log(10000.0) / embedding_dims))
        pe = torch.zeros(max_len, 1, embedding_dims)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class Dual_Computation_Block(nn.Module):
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm='ln',
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        **kwargs,
    ):
        super().__init__()
        '''
        super(Dual_Computation_Block, self).__init__(
            name='Dual_Computation_Block', **kwargs
        )
        '''
        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        self.norm = norm
        if norm is not None:
            self.intra_norm = nn.LayerNorm(normalized_shape = out_channels, eps=1e-8)
            self.inter_norm = nn.LayerNorm(normalized_shape = out_channels,eps=1e-8)

        if linear_layer_after_inter_intra:
            self.intra_linear = nn.Linear(in_features = out_channels, out_features = out_channels)
            self.inter_linear = nn.Linear(in_features = out_channels, out_features = out_channels)

    def forward(self, x): 
        B, N, K, S = x.shape
        
        intra = torch.reshape(torch.permute(x, (0, 3, 2, 1)), (B * S, K, N))
        intra = self.intra_mdl(intra) #(batch, seq, feature) B*S, K, N
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        intra = torch.reshape(intra, (B, S, K, N)) # B, S, K, N
        if self.norm is not None:
            intra = self.intra_norm(intra)
        intra = torch.permute(intra, (0, 2, 1, 3)) # B, K, S, N
        
        if self.skip_around_intra:
            intra = intra + torch.permute(x, (0, 2, 3, 1))

        '''
        intra = torch.permute(intra, (0, 3, 2, 1)) # B, N, K, S
        inter = torch.reshape(torch.permute(x, (0, 2, 3, 1)), (B * K, S, N))
        '''

        inter = torch.reshape(intra, (B * K, S, N))
        inter = self.inter_mdl(inter) #(batch, seq, feature) B*K, S, N
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        inter = torch.reshape(inter, (B, K, S, N)) # B, K, S, N
        if self.norm is not None:
            inter = self.inter_norm(inter)
        inter = torch.permute(inter, (0, 3, 1, 2)) # B, N, K, S

        #out = inter + intra # B, N, K, S
        if self.skip_around_intra:
            out = inter + x
        #out = inter
        return out


class Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
        activation=F.relu,
        **kwargs,
    ):  
        super().__init__()
        '''
        super(Dual_Path_Model, self).__init__(
            name='Dual_Path_Model', **kwargs
        )
        '''
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(normalized_shape=in_channels, eps=1e-8)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias=False)
        
        self.use_global_pos_enc = use_global_pos_enc
        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(embedding_dims = out_channels)

        self.conv2d = nn.Conv2d(out_channels, out_channels * num_spks, kernel_size=1)
        self.prelu = nn.PReLU()#shared_axes=[2, 3]) # TODO
        self.output_left = nn.Conv1d(out_channels, out_channels, kernel_size = 1)
        self.output_gate = nn.Conv1d(out_channels, out_channels, kernel_size = 1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, kernel_size = 1, bias=False)
        self.activation = activation
        self.out_channels = out_channels

        self.dual_mdl = nn.ModuleList([
                Dual_Computation_Block(
                    intra_model(),
                    inter_model(),
                    out_channels,
                    skip_around_intra=skip_around_intra,
                    linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                )
                for _ in range(num_layers)
        ])


    def forward(self, x):
        # B, N, L
        B, N, L = x.shape
        x = torch.permute(x, (0, 2, 1)) # B, L, N
        x = self.norm(x)
        x = torch.permute(x, (0, 2, 1)) # B, N, L
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x) + x * (N ** 0.5)
            
        # B, N, K, S
        x, gap = self._Segmentation(x, self.K)
        # B, N, K, S
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # B, N*spks, K, S
        x = self.conv2d(x)

        # B, N*spks, K, S
        B, _, K, S = x.shape

        # B*spks, N, L
        x = torch.reshape(x, (B * self.num_spks, -1, K, S))
        x = self._over_add(x, gap)

        # B*spks, N, L
        x = torch.tanh(self.output_left(x)) + torch.sigmoid(self.output_gate(x))
        x = self.end_conv1x1(x)

        _, N, L = x.shape
        x = torch.reshape(x, (B, self.num_spks, N, L))
        if self.activation:
            x = self.activation(x)
            
        # spks, B, N, L
        return torch.permute(x, (1, 0, 2, 3))

    def _padding(self, input, K):
        B, N, L = input.shape
        P = K // 2

        gap = K - (P + L % K) % K

        if gap > 0 :
            pad = torch.zeros(B, N, gap, device = input.device)
            input = torch.cat([input, pad], dim=2)

        _pad = torch.zeros(B, N, P, device = input.device)
        input = torch.cat([_pad, input, _pad], dim=2)
        return input, gap

    def _Segmentation(self, input, K):
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        input1 = torch.reshape(input[:, :, :-P], (B, N, -1, K))
        input2 = torch.reshape(input[:, :, P:], (B, N, -1, K))
        input = torch.cat([input1, input2], dim=3)
        input = torch.reshape(input, (B, N, -1, K))
        input = torch.permute(input, (0, 1, 3, 2))
        return input, gap

    def _over_add(self, input, gap):
        B, N, K, S = input.shape
        P = K // 2
        input = torch.reshape(torch.permute(input, (0, 1, 3, 2)), (B, N, -1, K * 2))
        input1 = torch.reshape(input[:, :, :, :K], (B, N, -1))[:, :, P:]
        input2 = torch.reshape(input[:, :, :, K:], (B, N, -1))[:, :, :-P]
        input = input1 + input2
        
        if gap > 0:
            input = input[:, :, :-gap]
        return input

class Model(nn.Module):
    def __init__(
        self,
        intra_model,
        inter_model,
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_useextralinearlayer=False,
        masknet_extraskipconnection=True,
        masknet_numspks=4,
        activation=F.relu,
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
                          nn.ReLU(),
                        )
        
        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            inter_model=inter_model,
            num_layers=masknet_numlayers,
            K=masknet_chunksize,
            num_spks=masknet_numspks,
            skip_around_intra=masknet_extraskipconnection,
            linear_layer_after_inter_intra=masknet_useextralinearlayer,
            activation=activation,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels = encoder_out_nchannels,
            out_channels=encoder_in_nchannels,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            bias=True,
        )

        self.num_spks = masknet_numspks

    def forward(self, mix):
        T_origin = mix.shape[2] # B, 1, L
        mix_w = self.encoder(mix) # B, N, L'
        est_mask = self.masknet(mix_w) # n_spks, B, N, L'
        mix_w = torch.tile(torch.unsqueeze(mix_w, dim = 1), (self.num_spks,1, 1, 1)) # n_spks, B, N, L'
        sep_h = mix_w * est_mask  # n_spks, B, N, L'

        n_spks, B, N, L = sep_h.shape
        sep_h = torch.reshape(sep_h, (n_spks* B, N, L)) # (n_spks * B, N, L')
        logits = self.decoder(sep_h) # (n_spks * B, 1, _L)
        estimate = torch.permute(torch.reshape(logits, (n_spks, B, -1)), [1,0,2]) # (B, n_spks, _L)
        
        T_est = estimate.shape[2]
        if  T_origin > T_est:
            estimate = F.pad(estimate, (0, T_origin - T_est,  0, 0, 0, 0), "constant", 0)
        else:
            estimate = estimate[:, :, :T_origin]
            
        return estimate
