"""Post-training references: can a model beat simple frequency guessing?"""
import hashlib
import json
import math
from pathlib import Path

import torch
from experiment import Config, Model, load_text, prepare_evaluation

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda")
train, valid, chars, metadata = load_text(device)
evaluation = prepare_evaluation(device, train, valid)
vocab = len(chars)
unigram_counts = torch.bincount(train, minlength=vocab).double() + 1
unigram = unigram_counts / unigram_counts.sum()
bigram_counts = torch.bincount(train[:-1] * vocab + train[1:], minlength=vocab*vocab).reshape(vocab, vocab).double() + 1
bigram = bigram_counts / bigram_counts.sum(1, keepdim=True)
unigram_bits, bigram_bits, n = 0., 0., 0
for x, y in evaluation["language"]:
    unigram_bits += (-unigram[y].log2()).sum().item()
    bigram_bits += (-bigram[x, y].log2()).sum().item()
    n += y.numel()
correct, count = 0, 0
for x, y in evaluation["lookup"]:
    values = x[:, 1:12:2] - 16
    hist = torch.nn.functional.one_hot(values, 8).sum(1)
    majority = hist.argmax(1)  # fixed lowest-value tie break, no query usage
    correct += (majority == y).sum().item()
    count += len(y)
x, target = evaluation["absolute"][0]
affine_best = ((target - target.mean()) ** 2).mean().item()
result = {"posthoc": "reference calculations only, no training or checkpoint selection",
          "language": {"uniform_bpc": math.log2(vocab), "train_unigram_bpc": unigram_bits/n,
                       "train_bigram_bpc": bigram_bits/n, "smoothing": "add-one; train split only",
                       "evaluated_character_targets": n},
          "lookup": {"uniform_accuracy": .125, "ignore_query_choose_most_frequent_value_accuracy": correct/count},
          "absolute": {"best_affine_grid_mse": affine_best}}
examples_x, examples_y = evaluation["lookup"][0]
examples = [{"pairs": [[int(k), int(v-16)] for k, v in zip(row[:12:2], row[1:12:2])],
             "query_key": int(row[-1]), "answer": int(y), "predictions": {}}
            for row, y in zip(examples_x[:4], examples_y[:4])]
for variant in ("standard", "attention", "attention_matched", "memory"):
    saved = torch.load(ROOT / "results/checkpoints" / f"lookup_{variant}_0.pt", map_location=device, weights_only=True)
    model = Model(Config(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["state_dict"])
    with torch.no_grad():
        predictions = model(examples_x[:4]).argmax(-1).tolist()
    for example, prediction in zip(examples, predictions):
        example["predictions"][variant] = prediction
result["lookup"]["first_four_fixed_examples_seed0"] = examples
protocol = json.loads((ROOT/"results/protocol.json").read_text())
assert hashlib.sha256((ROOT/"experiment.py").read_bytes()).hexdigest() == protocol["source_sha256"]
result["protocol_source_hash_verified"] = True
(ROOT/"diagnostics.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
