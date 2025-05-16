class CharTokenizer:
    def __init__(self, vocab):
        self.char2id = {ch: i + 2 for i, ch in enumerate(sorted(vocab))}
        self.char2id['<PAD>'] = 0
        self.char2id['<UNK>'] = 1
        self.id2char = {i: ch for ch, i in self.char2id.items()}

    def encode(self, text):
        return [self.char2id.get(ch, 1) for ch in text]

    def decode(self, ids):
        return ''.join([self.id2char.get(i, '?') for i in ids])

    def __len__(self):
        return len(self.char2id)
