import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import accuracy

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels).item()

    return total_loss / len(dataloader), total_acc / len(dataloader)
