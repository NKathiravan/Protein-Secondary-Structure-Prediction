import torch
import torch.nn as nn
from decoder_attention import MaskedMultiHeadAttention, CrossAttention
from feed_forward import FeedForward
from positional_encoding import PositionalEncoding
from layer_norm import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)
        
    def forward(self, x, encoder_output):
        self_attn_output, self_attn_weights = self.masked_attention(x)
        x = self.norm1(x + self_attn_output)
        cross_attn_output, cross_attn_weights = self.cross_attention(x, encoder_output)
        x = self.norm2(x + cross_attn_output)
        ff_output = self.ff(x)
        output = self.norm3(x + ff_output)
        return output, (self_attn_weights, cross_attn_weights)
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, num_layers):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
    def forward(self, x, encoder_output):
        seq_len = x.size(1)
        x = x + self.pos_encoding.forward(seq_len)
        attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, encoder_output)
            attn_weights.append(weights)
        return x, attn_weights
