# wrapper.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tokenizer import CharTokenizer
from data.dataset import ProteinDataset
from model.transformer import Transformer
from train import train
from config import config
from utils import accuracy

def run_training(
    data_path="proteinfolding/pssp_prediction/Transformer-2/data/TS115.csv",
    seq_len=350,
    d_model=128,
    n_heads=8,
    n_layers=1,
    batch_size=32,
    lr=1e-4,
    epochs=5,
    n_classes=8,
    save_path=None
):
    # 1. Load dataset
    df = pd.read_csv(data_path)

    # 2. Prepare tokenizer and label map
    amino_acids = set("".join(df["input"].values))
    labels = sorted(set("".join(df["dssp8"].values)))
    label_map = {ch: i for i, ch in enumerate(labels)}
    tokenizer = CharTokenizer(amino_acids)

    # 3. Dataset & Dataloader
    dataset = ProteinDataset(df, tokenizer, label_map, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        n_classes=n_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    # 5. Training Loop
    for epoch in range(epochs):
        loss, acc = train(model, dataloader, optimizer, loss_fn, device)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    # 6. Save model if needed
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model
