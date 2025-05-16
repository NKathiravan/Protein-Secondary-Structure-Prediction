import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

AA_VOCAB = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K':8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, '<pad>': 21, '<sos>': 22, '<eos>': 23}
Q8_VOCAB = {'G': 0, 'H': 1, 'I': 2, 'B': 3, 'E':4, 'S': 5, 'T': 6, 'C': 7, '<pad>':8, '<sos>':9, '<eos>': 10}

class PSSPDataset(Dataset):
    def __init__(self, file_path, max_len=100):
        data = pd.read_csv(file_path)
        self.sequences = data['sequence'].values
        self.labels = data['q8'].values
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        q8 = self.labels[idx]
        seq_ids = [AA_VOCAB['<sos>']] + [AA_VOCAB.get(aa, AA_VOCAB['X']) for aa in seq[:self.max_len]] + [AA_VOCAB['<eos>']]
        q8_ids = [Q8_VOCAB['<sos>']] + [Q8_VOCAB.get(label, Q8_VOCAB['C']) for label in q8[:self.max_len]] + [Q8_VOCAB['<eos>']]
        
        return seq_ids, q8_ids
    
def collate_fn(batch):
    seq_batch, q8_batch = zip(*batch)
    max_len = max(len(seq) for seq in seq_batch)
    seq_batch = [seq + [AA_VOCAB['<pad>']] * (max_len - len(seq)) for seq in seq_batch]
    q8_batch = [q8 + [Q8_VOCAB['<pad>']] * (max_len - len(q8)) for q8 in q8_batch]
    return torch.tensor(seq_batch), torch.tensor(q8_batch)

train_dataset = PSSPDataset('data/Train.csv')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
cb513_dataset = PSSPDataset('data/CB513.csv')
cb513_loader = DataLoader(cb513_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
ts115_dataset = PSSPDataset('data/TS115.csv')
ts115_loader = DataLoader(ts115_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
