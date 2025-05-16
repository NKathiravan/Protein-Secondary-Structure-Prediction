import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.transformer import Transformer
from tokenizer import CharTokenizer
from dataset import ProteinDataset
from train import train
from config import config

# Load and preprocess
df = pd.read_csv("proteinfolding/pssp_prediction/Transformer-2/data/TS115.csv")
amino_acids = set("".join(df["input"].values))
labels = set("".join(df["dssp8"].values))
label_map = {ch: i for i, ch in enumerate(sorted(labels))}

tokenizer = CharTokenizer(amino_acids)

dataset = ProteinDataset(df, tokenizer, label_map, seq_len=config["seq_len"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    vocab_size=len(tokenizer),
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    seq_len=config["seq_len"],
    n_classes=config["n_classes"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(config["epochs"]):
    loss, acc = train(model, dataloader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
