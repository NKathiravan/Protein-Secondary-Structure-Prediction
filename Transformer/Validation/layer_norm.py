import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        
    def forward(self, x):
        return self.norm(x)
