import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=4, d_ff=128, max_seq_len=100, num_layers=2,):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.encoder = Encoder(d_model, num_heads, d_ff, max_seq_len, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, max_seq_len, num_layers)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        # self.softmax = nn.Softmax(dim=-1)
        
    def create_padding_mask(self, x, pad_id):
        return (x != pad_id).unsqueeze(1).unsqueeze(2)    
        
    def forward(self, src, tgt, src_pad_id=0, tgt_pad_id=0):
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        src_mask = self.create_padding_mask(src, src_pad_id)
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_id)
        encoder_output, enc_attn_weights = self.encoder(src)
        decoder_output, dec_attn_weights = self.decoder(tgt, encoder_output)
        output = self.linear(decoder_output)
        # output = self.softmax(logits)
        return output, (enc_attn_weights, dec_attn_weights)
