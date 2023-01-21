import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable, Optional
from core.model.utils import _gen_positional_encoding
from core.model.encoderlayer import AdaptiveLayerNorm 
class MemoryTransformerEncoderLayer_MEMABSPOS_SINGLESHOT_AdaLN_MEMDROPOUT(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, depth:int, nhead: int, chunk_size:int, mem_size:int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert batch_first, "Batch first only"
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.chunk_size = chunk_size
        
        self.register_buffer('position_signal', _gen_positional_encoding(d_model, max_len = mem_size + chunk_size)) # 1, max_len, num_layers 
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = AdaptiveLayerNorm(d_model, depth = depth, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = AdaptiveLayerNorm(d_model, depth = depth, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super( MemoryTransformerEncoderLayer_MEMABSPOS_SINGLESHOT_AdaLN_MEMDROPOUT, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, seq: Tensor, mem: Tensor, depth:int, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, info = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.norm_first:
            B,L,N = seq.shape
            seq_mem = self.norm1(torch.cat((seq, mem), dim = 1), depth) # B, L+M, N
            seq_out, mem_out = self._sa_block(seq_mem[:,:L,:],seq_mem[:,L:,:], src_mask, src_key_padding_mask)
            seq, mem = seq + seq_out, mem + mem_out # (B, L, N), (B, M, N)
            
            seq_mem = self.norm2(torch.cat((seq, mem), dim = 1), depth) # B, L+M, N
            seq_mem = self._ff_block(seq_mem) # B, L+M, N
            seq, mem = seq + seq_mem[:, :L, :], mem + seq_mem[:, L:, :]# (B, L, N), (B, M, N)
        else:
            B,L,N = seq.shape
            seq_mem = torch.cat((seq, mem), dim = 1) # B, L+M, N
            seq_out, mem_out = self._sa_block(seq,mem, src_mask, src_key_padding_mask)
            seq_mem =  self.norm1(seq_mem + torch.cat((seq_out, mem_out), dim = 1), depth) # B, L+M, N
            seq_mem_out = self._ff_block(seq_mem)  # B, L+M, N
            seq_mem = self.norm2(seq_mem + seq_mem_out, depth)
            seq, mem = seq_mem[:, :L, :],seq_mem[:, L:, :] # (B, L, N), (B, M, N
        return seq, mem

    def _sa_block(self, x: Tensor, mem:Tensor, # x: B, L, N, mem: B, M, N
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], info = None) -> Tensor:
        
        seq, gap = self._overlap_chunk(x, self.chunk_size) # B, S, K, N        
        B, S, K, N = seq.shape
        
        seq = torch.reshape(seq, (B * S, K, N)) # B*S, K, N
        mem = mem.repeat((S,1,1)) # B*S, M, N
        memseq = torch.cat((mem,seq), dim = 1) + self.position_signal  # B*S, M+K, N
        memseq = self.self_attn(memseq, memseq, memseq,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0] # B*S, K, N
        seq = torch.reshape(memseq[:,-K:,:], (B,S,K,N)) # B,S,K,N
        mem = torch.reshape(memseq[:,:-K,:], (B, S, -1, N)) # B,S,M,N
        
        mem_out = torch.mean(mem, dim = 1) # B, M, N
        
        seq = self._overlap_add(seq,gap) # B, L, N
        
        return self.dropout1(seq), self.dropout3(mem_out)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def _padding(self, x, K):
        B, L, N = x.shape
        P = K // 2
        gap = K - (P + L % K) % K
        x = F.pad(x, (0,0, P, P+gap), "constant", 0)
        return x, gap

    def _overlap_chunk(self, input,K): # B, L, N
        B, L, N = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        input1 = torch.reshape(input[:, :-P, :], (B, -1, K, N))
        input2 = torch.reshape(input[:, P:, :], (B, -1, K, N))
        input = torch.cat([input1, input2], dim=2)
        input = torch.reshape(input, (B, -1, K, N)) # B, S, K, N
        #input = torch.permute(input, (0, 2, 3, 1)) # B, S, K, N
        return input, gap

    def _overlap_add(self, input, gap):
        B, S, K, N = input.shape
        P = K // 2
        #input = torch.reshape(torch.permute(input, (0, 3, 1, 2)), (B, N, -1, K * 2))
        input = torch.reshape(input, (B, -1, K * 2, N))
        input1 = torch.reshape(input[:, :, :K, :], (B, -1, N))[:, P:, :]
        input2 = torch.reshape(input[:, :, K:, :], (B, -1, N))[:, :-P, :]
        input = 0.5 * (input1 + input2) # B, L, N
        
        if gap > 0:
            input = input[:, :-gap, :]
        return input