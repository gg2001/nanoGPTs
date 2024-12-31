import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384  # C
n_head = 6
n_layer = 6
dropout = 0.2

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str):
    return [stoi[c] for c in s]


def decode(encoded: list):
    return "".join([itos[i] for i in encoded])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])
    return x.to(device), y.to(device)


class EntropyPatching:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def patch(self, byte_sequence):
        patches = []
        patch = []
        for byte in byte_sequence:
            entropy = torch.rand(1).item()  # Mock entropy calculation
            patch.append(byte)
            if entropy > self.threshold:
                patches.append(patch)
                patch = []
        if patch:
            patches.append(patch)
        return patches


entropy_patcher = EntropyPatching()


class LocalEncoder(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=dropout,
            ),
            num_layers=2,
        )

    def forward(self, x):
        return self.transformer(x)


class LatentTransformer(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=dropout,
            ),
            num_layers=n_layer,
        )

    def forward(self, patches):
        return self.layers(patches)


class LocalDecoder(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=dropout,
            ),
            num_layers=2,
        )

    def forward(self, latent_representations):
        return self.transformer(latent_representations)


class BLTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, n_embd)
        self.local_encoder = LocalEncoder(n_embd, n_head, block_size, dropout)
        self.latent_transformer = LatentTransformer(n_embd, n_head, n_layer, dropout)
        self.local_decoder = LocalDecoder(n_embd, n_head, dropout)
        self.output = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        embedded = self.encoder(x)
        patches = entropy_patcher.patch(embedded)
        local_encoded = torch.stack(
            [self.local_encoder(torch.tensor(p)) for p in patches]
        )
        latent_rep = self.latent_transformer(local_encoded)
        decoded = self.local_decoder(latent_rep)
        logits = self.output(decoded)
        return logits


model = BLTModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch("train")
    optimizer.zero_grad()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"Step {iter}, Loss: {loss.item():.4f}")
