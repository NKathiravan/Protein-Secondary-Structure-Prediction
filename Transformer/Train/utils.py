def accuracy(preds, labels):
    _, preds_max = preds.max(dim=2)
    correct = (preds_max == labels).float()
    mask = labels != 0
    correct = correct * mask
    return correct.sum() / mask.sum()
