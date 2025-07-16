import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Load data
with open('/home/husseinbitambuka/Dev/NLP/gpt-dev/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"The characters are: {''.join(chars)} (total: {vocab_size})")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
batch_size = 4
block_size = 8
max_iters = 300
eval_interval = 50
learning_rate = 1e-2
eval_iters = 200
n_embed = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data loading
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Loss estimation
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Instantiate model
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate sample
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)[0].tolist()
print("Generated text:")
print(decode(generated))
