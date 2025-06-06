import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd

lines = open('./input.txt', 'r').read()

vocab = sorted((set(lines)))

itos = {}
stoi = {}
for ch, i in enumerate(vocab):
    stoi[i] = ch
    itos[ch] = i


def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])

MASTER_CONFIG = {
    "vocab_size": len(vocab),
}

dataset = torch.tensor(encode(lines), dtype=torch.int8)

def get_batches(dataset, split, batch_size, context_length, config=MASTER_CONFIG):
    train = dataset[:int(len(dataset) * 0.8)]
    val = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test = dataset[int(len(dataset) * 0.9):]
    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test
    ix = torch.randint(0, batch_data.size(0) - context_length - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_length] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_length+1] for i in ix]).long()

    return x, y

MASTER_CONFIG.update({
    'batch_size': 8,
    'context_window': 16
})

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
[(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

@torch.no_grad()
def evaluate_loss(model, config = MASTER_CONFIG):
    out = {}
    for split in ['train', 'test']:
        losses = []
        for _ in range(10):
            x, y = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(x)
        losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

MASTER_CONFIG.update({
    'd_model': 128,
})

def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    losses = []
    start_time = time.time()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()

        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model)
            losses += [x]
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_lr())

    print("validation loss: ", losses[-1]['val'])
    return pd.DataFrame(losses).plot()

def generate(model, config=MASTER_CONFIG, max_new_tokens=30):
    idx = torch.zeros(5, 1).long()
    for _ in range(max_new_tokens):
        # call the model
        logits = model(idx[:, -config['context_window']:])
        last_time_step_logits = logits[
            :, -1, :
        ]  # all the batches (1), last time step, all the logits
        p = F.softmax(last_time_step_logits, dim=-1)  # softmax to get probabilities
        idx_next = torch.multinomial(
            p, num_samples=1
        )  # sample from the distribution to get the next token
        idx = torch.cat([idx, idx_next], dim=-1)  # append to the sequence
    return [decode(x) for x in idx.tolist()]

idx = torch.zeros(5, 1).long()

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(N) * frob norm
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

# dry run of sampling to understand
# import torch
# import torch.nn.functional as F

# batch_size = 2
# context_window = 3
# vocab_size = 4
# idx = torch.zeros(batch_size, 1).long()

# print("Initial idx:", idx)

# for step in range(3):
#     # Simulate fake logits from a model
#     logits = torch.rand(batch_size, context_window, vocab_size)

#     print(f"\nStep {step+1}")
#     print("Input to model:", idx[:, -context_window:])
#     print("Fake logits:\n", logits)

#     last_logits = logits[:, -1, :]  # shape: (B, V)
#     print("Last step logits:\n", last_logits)

#     probs = F.softmax(last_logits, dim=-1)
#     print("Softmax probs:\n", probs)

#     sampled = torch.multinomial(probs, num_samples=1)
#     print("Sampled next tokens:\n", sampled)

#     idx = torch.cat([idx, sampled], dim=-1)
#     print("Updated idx:\n", idx)

x = torch.tensor([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10]])

print(x[:, -3:])  # understand what it's doing
