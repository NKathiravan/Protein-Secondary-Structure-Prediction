import apt_pkg
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from preprocessing import train_loader, AA_VOCAB, Q8_VOCAB

src_vocab_size = len(AA_VOCAB)
tgt_vocab_size = len(Q8_VOCAB)
d_model = 64
num_heads = 4
d_ff = 128
max_seq_len = 100
num_layers = 2
epochs = 10
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, max_seq_len, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=Q8_VOCAB['<pad>'])
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous().view(-1)
            
            optimizer.zero_grad()
            output, _ = model(src, tgt_input, AA_VOCAB['<pad>'], Q8_VOCAB['<pad>'])
            loss = criterion(output.view(-1, tgt_vocab_size), tgt_output)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss : {total_loss/len(train_loader)}")
    
    torch.save(model.state_dict(), "model.pt")
    
if __name__ == "__main__":
    train()
