"""Verify published artifacts; optionally recompute all final checkpoint scores."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from experiment import Config, Model, evaluate, load_text, prepare_evaluation

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true", help="Recompute scores on fixed evaluation inputs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = ROOT / "SHA256SUMS"
    paths = []
    for row in manifest.read_text().splitlines():
        digest, name = row.split("  ", 1)
        path = ROOT / name
        assert path.is_file(), f"Missing: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Hash mismatch: {name}"
        paths.append(name)
    protocol = json.loads((ROOT / "results/protocol.json").read_text())
    assert hashlib.sha256((ROOT / "experiment.py").read_bytes()).hexdigest() == protocol["source_sha256"]
    raw = (ROOT / "data/tinyshakespeare.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol["data"]["sha256"]
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9"
    results = sorted((ROOT / "results").glob("*.json"))
    records = [json.loads(p.read_text()) for p in results if p.name != "protocol.json"]
    assert len(records) == 33
    expected = {(task, variant, seed)
                for task in ("absolute", "lookup", "language")
                for variant in ("standard", "attention", "attention_matched", "memory")
                for seed in (0, 1, 2)
                if (task, variant) != ("absolute", "attention_matched")}
    assert {(r["task"], r["variant"], r["seed"]) for r in records} == expected
    device = torch.device(args.device)
    evaluation = None
    if args.reload:
        train, valid, _, metadata = load_text(device)
        assert metadata["sha256"] == protocol["data"]["sha256"]
        evaluation = prepare_evaluation(device, train, valid)
    max_error = 0.
    for row in records:
        c = Config(**row["config"])
        filename = f"{row['task']}_{row['variant']}_{row['seed']}.pt"
        checkpoint_path = ROOT / "results/checkpoints" / filename
        assert checkpoint_path.relative_to(ROOT).as_posix() == row["checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["config"] == row["config"]
        assert checkpoint["seed"] == row["seed"]
        assert checkpoint["final"] == row["final"]
        assert row["final"]["step"] == protocol["tasks"][row["task"]]["steps"]
        assert row["settings"] == protocol["tasks"][row["task"]]
        model = Model(c)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == row["parameters"]
        if c.variant != "standard":
            assert not any("ffn" in name for name, _ in model.named_parameters())
        if args.reload:
            metrics = evaluate(model.to(device), evaluation[row["task"]], row["task"])
            error = abs(metrics["loss"] - row["final"]["loss"])
            assert error < 1e-4, (filename, error)
            if "accuracy" in metrics:
                assert abs(metrics["accuracy"] - row["final"]["accuracy"]) < 1e-3
            max_error = max(max_error, error)
    summary = json.loads((ROOT / "summary.json").read_text())
    for task, variants in summary.items():
        for variant, values in variants.items():
            rows = sorted([r for r in records if r["task"] == task and r["variant"] == variant], key=lambda r: r["seed"])
            numbers = [r["final"][values["metric"]] for r in rows]
            assert numbers == values["values"]
            assert abs(statistics.mean(numbers) - values["mean"]) < 1e-12
            assert abs(statistics.stdev(numbers) - values["sample_std"]) < 1e-12
    print(json.dumps({"files_verified": len(paths), "checkpoints_verified": len(records),
                      "original_training_source_hash": protocol["source_sha256"],
                      "reevaluated": args.reload, "device": str(device),
                      "maximum_loss_error": max_error if args.reload else None}, indent=2))


if __name__ == "__main__":
    main()
