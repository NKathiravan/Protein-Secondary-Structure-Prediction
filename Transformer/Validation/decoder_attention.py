import torch
import torch.nn as nn
from attention import MultiHeadAttention

class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads):
        super().__init__(d_model, num_heads)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len)).to(x.device)
        return super().forward(x, mask)

class CrossAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads):
        super().__init__(d_model, num_heads)
        
    def forward(self, x, encoder_output):
        batch_size, seq_len = x.size(0), x.size(1)
        src_seq_len = encoder_output.size(1)
    
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_output).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_output).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output, weights = self.attention(Q, K, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        return output, weights
