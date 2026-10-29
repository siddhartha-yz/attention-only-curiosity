"""Small, reproducible attention-only curiosity experiment; no pretrained weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class Config:
    task: str
    variant: str
    width: int = 64
    layers: int = 2
    heads: int = 4
    memory_slots: int = 0  # per head, only keys/values; no extra input/query tokens
    norms: bool = True
    vocab: int = 24
    context: int = 13


def configuration(task, variant, vocab=24):
    c = Config(task=task, variant=variant, vocab=vocab)
    if variant == "memory":
        c.memory_slots = 4 * c.width  # 2*M*d = 8*d^2, matching a 4d FFN
    elif variant == "attention_matched":
        c.layers = 6  # 6*4*d^2 ~= 2*12*d^2; embedding/readout are shared in size
    if task == "absolute":
        c.width, c.heads, c.layers = 32, 4, 2
        c.context, c.norms, c.vocab = 1, False, 0
        c.memory_slots = 4 * c.width if variant == "memory" else 0
    elif task == "language":
        c.context = 64
    return c


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dh, self.slots = c.heads, c.width // c.heads, c.memory_slots
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.output = nn.Linear(c.width, c.width, bias=False)
        if self.slots:
            self.memory_keys = nn.Parameter(torch.empty(c.heads, self.slots, self.dh))
            self.memory_values = nn.Parameter(torch.empty(c.heads, self.slots, self.dh))
        allowed = torch.ones(c.context, c.context, dtype=torch.bool).tril()
        if self.slots:
            allowed = torch.cat([allowed, torch.ones(c.context, self.slots, dtype=torch.bool)], 1)
        self.register_buffer("allowed", allowed, persistent=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        if self.slots:
            k = torch.cat([k, self.memory_keys.unsqueeze(0).expand(b, -1, -1, -1)], dim=2)
            v = torch.cat([v, self.memory_values.unsqueeze(0).expand(b, -1, -1, -1)], dim=2)
            # Memory always visible. Future sequence tokens remain masked.
            mask = torch.cat([self.allowed[:t, :t], self.allowed[:t, -self.slots:]], 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.width) if c.norms else nn.Identity()
        self.attention = Attention(c)
        self.ffn = None
        if c.variant == "standard":
            self.norm2 = nn.LayerNorm(c.width) if c.norms else nn.Identity()
            self.ffn = nn.Sequential(nn.Linear(c.width, 4 * c.width, bias=False),
                                     nn.GELU(), nn.Linear(4 * c.width, c.width, bias=False))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        if c.task == "absolute":
            self.input = nn.Linear(1, c.width)
        else:
            self.embedding = nn.Embedding(c.vocab, c.width)
            self.position = nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.final_norm = nn.LayerNorm(c.width) if c.norms else nn.Identity()
        out = 1 if c.task == "absolute" else (8 if c.task == "lookup" else c.vocab)
        self.readout = nn.Linear(c.width, out, bias=c.task == "absolute")

    def initialize(self, seed):
        # Names, rather than parameter traversal order, determine RNG seeds.
        # Shared same-shape components start identically across variants.
        for name, p in self.named_parameters():
            digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
            g = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
            if "norm" in name:
                nn.init.ones_(p) if name.endswith("weight") else nn.init.zeros_(p)
            elif "memory_" in name:
                nn.init.normal_(p, std=1.0, generator=g)
            elif "embedding" in name or "position" in name:
                nn.init.normal_(p, std=0.1, generator=g)
            elif p.ndim >= 2:
                bound = 1 / math.sqrt(p.shape[-1])
                nn.init.uniform_(p, -bound, bound, generator=g)
            else:
                nn.init.zeros_(p)
        return self

    def forward(self, x):
        if self.config.task == "absolute":
            h = self.input(x)
        else:
            positions = torch.arange(x.shape[1], device=x.device)
            h = self.embedding(x) + self.position(positions)
        for block in self.blocks:
            h = block(h)
        y = self.readout(self.final_norm(h))
        return y[:, -1] if self.config.task in ("absolute", "lookup") else y


def lookup_batch(batch, generator, device):
    # Unique keys, independently random values, random queried pair.
    keys = torch.rand(batch, 16, generator=generator, device=device).argsort(1)[:, :6]
    values = torch.randint(8, (batch, 6), generator=generator, device=device)
    which = torch.randint(6, (batch, 1), generator=generator, device=device)
    tokens = torch.empty(batch, 13, dtype=torch.long, device=device)
    tokens[:, :12:2] = keys
    tokens[:, 1:12:2] = values + 16
    tokens[:, -1] = keys.gather(1, which).squeeze(1)
    labels = values.gather(1, which).squeeze(1)
    return tokens, labels


def language_batch(data, batch, context, generator):
    starts = torch.randint(len(data) - context, (batch,), device=data.device, generator=generator)
    positions = starts[:, None] + torch.arange(context, device=data.device)[None, :]
    return data[positions], data[positions + 1]


def load_text(device):
    path = ROOT / "data" / "tinyshakespeare.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            path.write_bytes(response.read())
    text = path.read_text()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long, device=device)
    split = int(len(encoded) * 0.9)
    metadata = {"url": DATA_URL, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "characters": len(text), "vocabulary": chars, "train_characters": split,
                "validation_characters": len(encoded) - split,
                "split": "contiguous first 90% train; final 10% validation; no crossing windows"}
    return encoded[:split], encoded[split:], chars, metadata


def prepare_evaluation(device, train, valid):
    g = torch.Generator(device=device).manual_seed(20261028)
    x = torch.linspace(-1, 1, 1001, device=device).view(-1, 1, 1)
    result = {"absolute": [(x, x[:, 0].abs())],
              "lookup": [lookup_batch(256, g, device) for _ in range(16)],
              "language": [language_batch(valid, 32, 64, g) for _ in range(32)],
              "language_train": [language_batch(train, 32, 64, g) for _ in range(8)]}
    return result


@torch.no_grad()
def evaluate(model, batches, task):
    model.eval()
    total_loss, correct, count = 0., 0, 0
    for x, y in batches:
        pred = model(x)
        if task == "absolute":
            total_loss += F.mse_loss(pred, y, reduction="sum").item()
            count += y.numel()
        else:
            total_loss += F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1), reduction="sum").item()
            correct += (pred.argmax(-1) == y).sum().item()
            count += y.numel()
    model.train()
    result = {"loss": total_loss / count, "n": count}
    if task != "absolute":
        result["accuracy"] = correct / count
    if task == "language":
        result["bits_per_character"] = result["loss"] / math.log(2)
    return result


@torch.no_grad()
def generate(model, chars, device, seed=777, length=240):
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    tokens = [stoi[c] for c in "ROMEO:\n"]
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(length):
        x = torch.tensor([tokens[-model.config.context:]], device=device)
        logits = model(x)[0, -1] / 0.8
        tokens.append(torch.multinomial(logits.softmax(-1), 1, generator=g).item())
    return "".join(chars[i] for i in tokens)


def check_invariants(device):
    torch.manual_seed(123)
    for variant in ("standard", "attention", "attention_matched", "memory"):
        model = Model(configuration("lookup", variant)).initialize(0).to(device).eval()
        g = torch.Generator(device=device).manual_seed(123)
        x, labels = lookup_batch(16, g, device)
        for i in range(len(x)):
            p = (x[i, :12:2] == x[i, -1]).nonzero().item()
            assert labels[i].item() == x[i, 2 * p + 1].item() - 16
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 24
        with torch.no_grad():
            def hidden(z):
                h = model.embedding(z) + model.position(torch.arange(13, device=device))
                for block in model.blocks:
                    h = block(h)
                return h
            causal_error = (hidden(x)[:, :8] - hidden(changed)[:, :8]).abs().max().item()
        assert causal_error < 1e-6, (variant, causal_error)
        if variant == "memory":
            F.cross_entropy(model(x), labels).backward()
            assert all(p.grad is not None and p.grad.norm() > 0 for n, p in model.named_parameters() if "memory_" in n)
        else:
            assert not any("ffn" in n for n, _ in model.named_parameters()) if variant != "standard" else True
    model = Model(configuration("absolute", "attention")).initialize(0).to(device).eval()
    x = torch.tensor([[[-1.]], [[0.]], [[1.]]], device=device)
    with torch.no_grad():
        y = model(x).squeeze()
    affine_error = abs((y[0] + y[2]).item() / 2 - y[1].item())
    assert affine_error < 1e-5, affine_error
    a = Model(configuration("lookup", "standard")).initialize(0)
    b = Model(configuration("lookup", "attention")).initialize(0)
    for name, p in b.named_parameters():
        assert torch.equal(p, dict(a.named_parameters())[name]), name
    return {"causal_mask": "pass for all four variants, including memory visibility",
            "lookup_target": "pass", "memory_gradients": "pass",
            "shared_initialization": "pass", "single_token_affine_midpoint_error": affine_error}


def run_one(task, variant, seed, protocol, data, evaluation, out):
    train, _, chars = data
    c = configuration(task, variant, len(chars) if task == "language" else 24)
    device = train.device
    model = Model(c).initialize(seed).to(device)
    params = sum(p.numel() for p in model.parameters())
    settings = protocol["tasks"][task]
    steps, batch, lr = settings["steps"], settings["batch"], settings["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01 if task != "absolute" else 0.0)
    g = torch.Generator(device=device).manual_seed(10000 + seed)
    torch.cuda.reset_peak_memory_stats()
    history = []
    started = time.perf_counter()
    train_seconds = 0.
    for step in range(steps + 1):
        if step % settings["evaluate_every"] == 0 or step == steps:
            metrics = evaluate(model, evaluation[task], task)
            row = {"step": step, "train_seconds": train_seconds, **metrics}
            history.append(row)
            print(json.dumps({"task": task, "variant": variant, "seed": seed, **row}), flush=True)
        if step == steps:
            break
        torch.cuda.synchronize()
        tick = time.perf_counter()
        if task == "absolute":
            x = torch.rand(batch, 1, 1, device=device, generator=g) * 2 - 1
            y = x[:, 0].abs()
        elif task == "lookup":
            x, y = lookup_batch(batch, g, device)
        else:
            x, y = language_batch(train, batch, c.context, g)
        warmup = min(50, max(1, steps // 10))
        fraction = max(0., (step - warmup) / max(1, steps - warmup))
        factor = min(1., (step + 1) / warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y) if task == "absolute" else F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss: {task}/{variant}/{seed}/{step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - tick
    result = {"task": task, "variant": variant, "seed": seed, "config": asdict(c),
              "parameters": params, "settings": settings, "history": history,
              "final": history[-1], "training_seconds": train_seconds,
              "wall_seconds": time.perf_counter() - started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
    if task == "absolute":
        with torch.no_grad():
            x = evaluation[task][0][0]
            result["curve"] = {"x": x.flatten().tolist(), "prediction": model(x).flatten().tolist()}
    elif task == "language":
        result["train_probe"] = evaluate(model, evaluation["language_train"], task)
        result["sample"] = generate(model, chars, device)
    checkpoint = out / "checkpoints" / f"{task}_{variant}_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(c), "state_dict": {n: p.cpu() for n, p in model.state_dict().items()},
                "seed": seed, "final": result["final"]}, checkpoint)
    reloaded = Model(c).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded.load_state_dict(saved["state_dict"])
    reload_metrics = evaluate(reloaded, evaluation[task], task)
    assert abs(reload_metrics["loss"] - result["final"]["loss"]) < 1e-6
    result["checkpoint"] = str(checkpoint.relative_to(ROOT))
    result["reload_loss_error"] = abs(reload_metrics["loss"] - result["final"]["loss"])
    (out / f"{task}_{variant}_{seed}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"completed": f"{task}/{variant}/{seed}", "parameters": params,
                      "final": result["final"], "training_seconds": train_seconds}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task", choices=["absolute", "lookup", "language", "all"], default="all")
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variant", nargs="+", default=["standard", "attention", "attention_matched", "memory"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot is configured for the available local CUDA GPU.")
    device = torch.device("cuda")
    train, valid, chars, metadata = load_text(device)
    evaluation = prepare_evaluation(device, train, valid)
    out = ROOT / ("smoke" if args.smoke else "results")
    out.mkdir(exist_ok=True)
    protocol = {"date": "2026-10-28", "purpose": "curiosity, not a paper reproduction or scale claim",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seeds": [0, 1, 2], "precision": "float32; TF32 allowed; no AMP, no dropout",
                "tasks": {
                    "absolute": {"steps": 400, "batch": 256, "learning_rate": 0.002, "evaluate_every": 100},
                    "lookup": {"steps": 1000, "batch": 128, "learning_rate": 0.001, "evaluate_every": 200},
                    "language": {"steps": 800, "batch": 32, "learning_rate": 0.0015, "evaluate_every": 200}},
                "comparison": "same steps, batches and data RNG per seed; standard vs 6-layer attention approximately parameter matched, not compute matched",
                "selection": "final checkpoint only; no best-validation checkpoint selection; identical hyperparameters per task across variants",
                "data": metadata, "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda},
                "sources": [DATA_URL, "https://arxiv.org/abs/1907.01470", "https://proceedings.mlr.press/v139/dong21a.html"],
                "invariants": check_invariants(device)}
    if args.smoke:
        for setting in protocol["tasks"].values():
            setting.update(steps=20, evaluate_every=20)
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, "Protocol changed; use a separate result directory."
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    tasks = ["absolute", "lookup", "language"] if args.task == "all" else [args.task]
    for task in tasks:
        for seed in args.seed:
            # Rotate execution order to reduce systematic temperature/order effects.
            variants = args.variant[seed % len(args.variant):] + args.variant[:seed % len(args.variant)]
            for variant in variants:
                if task == "absolute" and variant == "attention_matched":
                    continue
                result_path = out / f"{task}_{variant}_{seed}.json"
                if result_path.exists():
                    print(f"Already complete: {result_path.name}", flush=True)
                    continue
                run_one(task, variant, seed, protocol, (train, valid, chars), evaluation, out)


if __name__ == "__main__":
    main()
