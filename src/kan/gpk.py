"""
    Generative Pretrained KANsformer
"""
import os

import urllib
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class Tokenizer:
    def __init__(self, text: str):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)

        self.string_to_integers = { ch:i for i, ch in enumerate(self.vocab)}
        self.intergers_to_string = { i:ch for i, ch in enumerate(self.vocab)}

    def encode(self, s: str) -> list[int]:
        return [self.string_to_integers[c] for c in s]

    def decode(self, l: int) -> list[str]:
        return "".join([self.intergers_to_string[i] for i in l])

class DataLoader:
    def __init__(
        self,
        data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device="cpu",
    ):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))

        x = torch.stack([self.data[i:i+self.block_size] for i in ix]).to(self.device)
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix]).to(self.device)

        return x, y

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            return logits
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        device="cpu",
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if targets is None:
            return logits
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class GPK(nn.Module):
    def __init__(self):
        super().__init__()

@torch.no_grad()
def estimate_loss(
    model: BigramModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    eval_iters: int,
):
    out = {}
    model.eval()

    for split, loader in [("train", train_loader), ("test", test_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.backends.openmp.is_available():
        device = "opencl"
    else:
        device = "cpu"

    torch.device(device)
    torch.manual_seed(1337)

    split_ratio = 0.8
    batch_size = 32
    block_size = 8
    epochs = 5
    max_iters = 5000
    eval_interval = 200
    learning_rate = 1e-2
    eval_iters = 500
    n_embd = 32

    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = "./data/"
    file_name = "input.txt"
    file_path = data_path + file_name

    if not os.path.exists(data_path):
        os.mkdir(data_path)
        urllib.request.urlretrieve(data_url, file_path)

    with open(file_path, mode="r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text=text)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(split_ratio * len(data))
    train_data = data[:n]
    test_data = data[n:]

    train_loader = DataLoader(train_data, block_size, batch_size)
    test_loader = DataLoader(test_data, block_size, batch_size)

    model = BigramModel(tokenizer.vocab_size)
    # model = GPT(tokenizer.vocab_size, block_size, n_embd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for i in tqdm(range(max_iters)):
        if i % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_loader,
                test_loader,
                eval_iters,
            )
            print(f'Step: {i}\nTrain Loss: {losses["train"]:.4f}\nTest Loss: {losses["test"]:.4f}')

        xb, yb = train_loader.get_batch()

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    context = torch.zeros((1, 1), dtype=torch.long)
    result = model.generate(context, max_new_tokens=500)[0].tolist()
    decoded_result = tokenizer.decode(result)

    print(decoded_result)