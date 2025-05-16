import torch
from transformer import Transformer
from preprocessing import cb513_loader, ts115_loader, AA_VOCAB, Q8_VOCAB

def decode(model, src, max_len, device, src_pad_id, tgt_pad_id, sos_id, eos_id):
    model.eval()
    src = torch.tensor([src], dtype=torch.long).to(device)
    
    with torch.no_grad():
        encoder_output, _ = model.encoder(model.src_embedding(src), model.create_padding_mask(src, src_pad_id))
        
    tgt = torch.tensor([[sos_id]], dtype=torch.long).to(device)
    for _ in range(max_len):
        with torch.no_grad():
            output, _ = model.decoder(model.tgt_embedding(tgt), encoder_output)
            logits = model.linear(output[:, -1])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == eos_id:
            break
    return tgt[0].cpu().numpy()

def evaluate(loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            for i in range(src.size(0)):
                pred = decode(model, src[1].cpu().numpy(), max_len=src.size(1)+2, device=device, src_pad_id=AA_VOCAB['<pad>'], tgt_pad_id=Q8_VOCAB['<pad>'],
                              sos_id=Q8_VOCAB['<sos>'], eos_id=Q8_VOCAB['<eos>'])
                true = tgt[i].cpu().numpy()
                for p, t in zip(pred[1:], true[1:]):
                    if t != Q8_VOCAB['<pad>'] and t != Q8_VOCAB['<eos>']:
                        total += 1
                        if p == t:
                            correct += 1
    return correct/total if total > 0 else 0

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(len(AA_VOCAB), len(Q8_VOCAB), d_model=64, num_heads=4, d_ff=128, max_seq_len=100, num_layers=2)
    model.load_state_dict(torch.load('model.pt'))
    model.to(device)
    
    cb513_acc = evaluate(cb513_loader, model, device)
    ts115_acc = evaluate(ts115_loader, model, device)
    print(f"CB513 Q8 Accuracy: {cb513_acc:.4f}")
    print(f"TS115 Q8 Accuracy: {ts115_acc:.4f}")
    
if __name__ == "__main__":
    test()
