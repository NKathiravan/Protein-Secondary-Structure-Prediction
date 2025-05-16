import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.functional import pad
from tqdm import tqdm
from io import StringIO
from sklearn.metrics import precision_score, recall_score, f1_score

# Setting up quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Simplified quantum self-attention circuit
@qml.qnode(dev, interface="torch")
def quantum_attention_circuit(q, k, params):
    # Encode query and key
    for i in range(n_qubits):
        qml.RX(q[i], wires=i)
        qml.RZ(k[i], wires=i)
    # Single variational layer
    for i in range(n_qubits):
        qml.RX(params[i], wires=i)
        qml.CNOT(wires=[i, (i+1) % n_qubits])
    # Measure single observable
    return qml.expval(qml.PauliZ(0))

# Mapping amino acids and DSSP3 labels
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
dssp3_labels = ['H', 'E', 'C']
dssp3_to_idx = {label: idx for idx, label in enumerate(dssp3_labels)}

# Data preprocessing
def preprocess_data(csv_data, max_len=64):  # Reduced max_len for faster testing
    df = pd.read_csv(StringIO(csv_data))
    sequences = df['input'].tolist()
    dssp3 = df['dssp3'].tolist()
    
    X, y = [], []
    for seq, labels in zip(sequences, dssp3):
        seq_idx = [aa_to_idx.get(aa, 0) for aa in seq[:max_len]]
        if len(seq_idx) < max_len:
            seq_idx = seq_idx + [0] * (max_len - len(seq_idx))
        label_idx = [dssp3_to_idx.get(label, 2) for label in labels[:max_len]]
        if len(label_idx) < max_len:
            label_idx = label_idx + [2] * (max_len - len(label_idx))
        X.append(seq_idx)
        y.append(label_idx)
    
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Quantum Self-Attention Layer
class QuantumSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(QuantumSelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Linear layers for query, key, value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Simplified quantum parameters (one RX angle per qubit)
        self.attn_params = nn.Parameter(torch.randn(n_qubits) * 0.1)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.nhead, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.nhead, self.d_k)
        
        attn_scores = torch.zeros(batch_size, self.nhead, seq_len, seq_len, device=x.device)
        
        # Progress tracking for quantum attention
        total_ops = batch_size * self.nhead * seq_len * seq_len
        op_count = 0
        print(f"Starting quantum attention computation ({total_ops} operations)...")
        
        for b in range(batch_size):
            for h in range(self.nhead):
                for i in range(seq_len):
                    for j in range(seq_len):
                        q = Q[b, i, h, :n_qubits]
                        k = K[b, j, h, :n_qubits]
                        q = (q - q.min()) / (q.max() - q.min() + 1e-8) * np.pi
                        k = (k - k.min()) / (k.max() - k.min() + 1e-8) * np.pi
                        score = quantum_attention_circuit(q, k, self.attn_params)
                        attn_scores[b, h, i, j] = score
                        
                        # Update progress
                        op_count += 1
                        if op_count % (total_ops // 10 or 1) == 0:
                            print(f"Quantum attention: {op_count}/{total_ops} operations ({op_count/total_ops*100:.1f}%)")
        
        attn_weights = torch.softmax(attn_scores / np.sqrt(self.d_k), dim=-1)
        V = V.transpose(1, 2)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        return output

# Custom Transformer Encoder Layer
class QuantumTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super(QuantumTransformerEncoderLayer, self).__init__()
        self.self_attn = QuantumSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Quantum-Hybrid Transformer Model
class QuantumTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=64, num_classes=3):
        super(QuantumTransformer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        self.transformer_encoder = nn.ModuleList([
            QuantumTransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        embeddings = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        embeddings = embeddings + self.pos_encoder[:, :seq_len, :]
        
        output = embeddings
        for layer in self.transformer_encoder:
            output = layer(output)
        
        output = self.fc(output)
        return output

# Training function with progress bar


def train_model(model, X, y, epochs=25, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        batch_iterator = tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for i in batch_iterator:
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_y.view(-1))
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=-1)

            # Flatten and collect predictions and labels
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(batch_y.view(-1).cpu().numpy())

            epoch_loss += loss.item()
            total_samples += batch_y.numel()

            correct = (preds == batch_y).sum().item()
            batch_iterator.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Batch Acc": f"{correct / batch_y.numel():.4f}"
            })

        # Remove padding index (if any) for evaluation - here it's label 2 ("C") assumed as default filler
        valid_indices = [i for i, label in enumerate(all_labels) if label != 2]
        valid_preds = [all_preds[i] for i in valid_indices]
        valid_labels = [all_labels[i] for i in valid_indices]

        precision = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
        recall = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
        f1 = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)

        avg_loss = epoch_loss / len(batch_iterator)
        print(f"Epoch {epoch+1}/{epochs} Summary: Avg Loss: {avg_loss:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Main execution
if __name__ == "__main__":
    # Load data
    with open("TS115.csv", "r") as f:
        csv_data = f.read()
    
    X, y = preprocess_data(csv_data)
    
    # Initialize model
    model = QuantumTransformer(vocab_size=len(amino_acids), d_model=64, nhead=4, num_layers=2, max_len=64, num_classes=len(dssp3_labels))
    
    # Train model
    train_model(model, X, y)
    
    # Example inference
    model.eval()
    with torch.no_grad():
        test_seq = X[:1]
        pred = model(test_seq)
        pred_labels = torch.argmax(pred, dim=-1)
        print("Predicted DSSP3 labels:", [dssp3_labels[idx] for idx in pred_labels[0].tolist()])