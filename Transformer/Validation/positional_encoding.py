import torch
import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_seq_len):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = self._create_positional_encoding()
        
    def _create_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float)*(-torch.log(torch.tensor(10000.0)) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, seq_len):
        assert seq_len <= self.max_seq_len
        return self.pe[:, :seq_len, :]
    
