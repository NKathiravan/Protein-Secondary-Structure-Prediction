import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.functional import pad
from io import StringIO
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Quantum setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_encoding(inputs):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Encoding and label maps
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
dssp3_labels = ['H', 'E', 'C']
dssp3_to_idx = {label: idx for idx, label in enumerate(dssp3_labels)}

def preprocess_data(csv_data, max_len=128):
    df = pd.read_csv(StringIO(csv_data))
    sequences = df['input'].tolist()
    dssp3 = df['dssp3'].tolist()

    X, y = [], []
    for seq, labels in zip(sequences, dssp3):
        seq_idx = [aa_to_idx.get(aa, 0) for aa in seq[:max_len]]
        label_idx = [dssp3_to_idx.get(label, 2) for label in labels[:max_len]]

        seq_idx += [0] * (max_len - len(seq_idx))
        label_idx += [2] * (max_len - len(label_idx))

        X.append(seq_idx)
        y.append(label_idx)

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class QuantumTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=128, num_classes=3):
        super(QuantumTransformer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.quantum_projection = nn.Linear(n_qubits, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape
        embeddings = []
        for i in range(batch_size):
            seq_emb = []
            for j in range(min(seq_len, self.max_len)):
                aa_idx = x[i, j].item()
                angles = torch.tensor([aa_idx * np.pi / 20] * n_qubits, dtype=torch.float)
                q_out = quantum_encoding(angles)
                seq_emb.append(torch.tensor(q_out, dtype=torch.float))
            seq_emb = torch.stack(seq_emb)
            if seq_len < self.max_len:
                seq_emb = pad(seq_emb, (0, 0, 0, self.max_len - seq_len))
            embeddings.append(seq_emb)

        embeddings = torch.stack(embeddings)
        embeddings = self.quantum_projection(embeddings)
        embeddings = embeddings * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        embeddings = embeddings + self.pos_encoder[:, :embeddings.size(1), :]

        output = self.transformer_encoder(embeddings)
        return self.fc(output)

def train_model(model, X, y, epochs=25, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = 0

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
            correct = (preds == batch_y).sum().item()
            total = batch_y.numel()

            epoch_loss += loss.item()
            epoch_correct += correct
            total_samples += total

            all_preds.append(preds.view(-1).cpu())
            all_targets.append(batch_y.view(-1).cpu())

            batch_iterator.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Batch Acc": f"{correct/total:.4f}"
            })

        avg_loss = epoch_loss / len(batch_iterator)
        avg_acc = epoch_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} Summary: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")

    # Compute overall metrics
    preds_flat = torch.cat(all_preds).numpy()
    targets_flat = torch.cat(all_targets).numpy()

    precision = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)

    print(f"\n--- Final Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    with open("TS115.csv", "r") as f:
        csv_data = f.read()

    X, y = preprocess_data(csv_data)
    model = QuantumTransformer(vocab_size=len(amino_acids), d_model=64, nhead=4, num_layers=2, max_len=128, num_classes=len(dssp3_labels))

    train_model(model, X, y)

    model.eval()
    with torch.no_grad():
        test_seq = X[:1]
        pred = model(test_seq)
        pred_labels = torch.argmax(pred, dim=-1)
        print("Predicted DSSP3 labels:", [dssp3_labels[idx] for idx in pred_labels[0].tolist()])
