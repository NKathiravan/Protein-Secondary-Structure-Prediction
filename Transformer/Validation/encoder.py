import torch 
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from positional_encoding import PositionalEncoding
from layer_norm import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model,  d_ff)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        output = self.norm2(x + ff_output)
        return output, attn_weights
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, num_layers):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = x + self.pos_encoding.forward(seq_len)
        attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            attn_weights.append(weights)
        return x, attn_weights
