import torch
import torch.nn as nn
from torch.nn import functional as F

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


def encode(s: str) -> list[int]:
    """Encoder: take a string, output a list of integers"""
    return [stoi[c] for c in s]


def decode(encoded: list[int]) -> str:
    """Decoder: take a list of integers, output a string"""
    return "".join([itos[i] for i in encoded])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a small batch of data of inputs x and targets y"""
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        # D: embedding dimension, H: number of heads, K: key/value size per head
        assert n_embd % n_head == 0
        self.H, self.K = n_head, n_embd // n_head
        self.qkv_DHD = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj_DD = nn.Linear(n_embd, n_embd)
        # register attention mask
        self.register_buffer("mask_TT", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        B, T, D = x_BTD.shape

        # compute query, key, value vectors
        qkv_BT3D = self.qkv_DHD(x_BTD)
        q_BTD, k_BTD, v_BTD = qkv_BT3D.split(D, dim=-1)

        # split heads and transpose to get B H T K
        q_BHTK = q_BTD.view(B, T, self.H, self.K).transpose(1, 2)
        k_BHTK = k_BTD.view(B, T, self.H, self.K).transpose(1, 2)
        v_BHTK = v_BTD.view(B, T, self.H, self.K).transpose(1, 2)

        # compute attention scores
        attn_BHTT = q_BHTK @ k_BHTK.transpose(-2, -1) * self.K**-0.5
        attn_BHTT = attn_BHTT.masked_fill(self.mask_TT[:T, :T] == 0, float("-inf"))
        attn_BHTT = F.softmax(attn_BHTT, dim=-1)
        attn_BHTT = self.dropout(attn_BHTT)

        # apply attention to values
        out_BHTK = attn_BHTT @ v_BHTK

        # merge heads and project
        out_BTD = out_BHTK.transpose(1, 2).reshape(B, T, D)
        out_BTD = self.dropout(self.proj_DD(out_BTD))

        return out_BTD


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # D: embedding dimension, F: FFN hidden dimension
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        return self.net(x_BTD)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        x_BTD = x_BTD + self.sa(self.ln1(x_BTD))
        x_BTD = x_BTD + self.ffwd(self.ln2(x_BTD))
        return x_BTD


class NanoGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ):
        super().__init__()
        # V: vocab size, T: sequence length (block size), D: embedding dimension
        self.token_embedding_VD = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_TD = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head_DV = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx_BT: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx_BT.shape
        device = idx_BT.device

        # token and position embeddings
        tok_emb_BTD = self.token_embedding_VD(idx_BT)
        pos_emb_TD = self.position_embedding_TD(torch.arange(T, device=device))
        x_BTD = tok_emb_BTD + pos_emb_TD

        # transformer blocks and final layer norm
        x_BTD = self.blocks(x_BTD)
        x_BTD = self.ln_f(x_BTD)
        logits_BTV = self.lm_head_DV(x_BTD)

        # compute loss if targets provided
        if targets is None:
            loss = None
        else:
            logits_NV = logits_BTV.view(-1, logits_BTV.size(-1))
            targets_N = targets.view(-1)
            loss = F.cross_entropy(logits_NV, targets_N)

        return logits_BTV, loss

    def generate(self, idx_BT: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # crop context if needed
            idx_cond = idx_BT[:, -block_size:]
            # get predictions
            logits_BTV, _ = self(idx_cond)
            # focus on last time step
            logits_BV = logits_BTV[:, -1, :]
            # sample from distribution
            probs_BV = F.softmax(logits_BV, dim=-1)
            idx_next_B1 = torch.multinomial(probs_BV, num_samples=1)
            # append to sequence
            idx_BT = torch.cat((idx_BT, idx_next_B1), dim=1)
        return idx_BT


model = NanoGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
