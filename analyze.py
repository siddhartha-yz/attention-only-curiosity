"""Create plots and machine-readable aggregates from completed final checkpoints."""
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VARIANTS = ["standard", "attention", "attention_matched", "memory"]
LABELS = {"standard": "标准 Transformer", "attention": "纯 Attention（2 层）",
          "attention_matched": "纯 Attention（6 层，参数接近）", "memory": "Attention＋可学习记忆"}
COLORS = {"standard": "#237a57", "attention": "#d66a28", "attention_matched": "#376ca5", "memory": "#9660aa"}
cjk_fonts = [p for p in font_manager.findSystemFonts() if "NotoSansCJK" in Path(p).name]
font_family = "sans-serif"
if cjk_fonts:
    font_manager.fontManager.addfont(cjk_fonts[0])
    font_family = font_manager.FontProperties(fname=cjk_fonts[0]).get_name()
plt.rcParams.update({"font.family": font_family, "font.size": 11, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fafaf8",
                     "axes.facecolor": "#fafaf8", "savefig.facecolor": "#fafaf8"})


def main():
    rows = [json.loads(p.read_text()) for p in RESULTS.glob("*.json") if p.name != "protocol.json"]
    rows = [r for r in rows if "task" in r]
    assert len(rows) == 33, f"Expected 33 final runs, got {len(rows)}"
    summary = {}
    for task in ("absolute", "lookup", "language"):
        summary[task] = {}
        for variant in VARIANTS:
            runs = sorted([r for r in rows if r["task"] == task and r["variant"] == variant], key=lambda r: r["seed"])
            if not runs:
                continue
            assert [r["seed"] for r in runs] == [0, 1, 2]
            key = "accuracy" if task == "lookup" else ("bits_per_character" if task == "language" else "loss")
            values = [r["final"][key] for r in runs]
            summary[task][variant] = {"metric": key, "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values), "values": values,
                "parameters": runs[0]["parameters"],
                "training_seconds_mean": statistics.mean(r["training_seconds"] for r in runs),
                "peak_allocated_mib_max": max(r["peak_allocated_mib"] for r in runs),
                "reload_loss_error_max": max(r["reload_loss_error"] for r in runs)}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    with (ROOT / "runs.csv").open("w") as f:
        names = ["task", "variant", "seed", "parameters", "loss", "accuracy", "bits_per_character", "training_seconds", "peak_allocated_mib"]
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, r["final"].get(k, "")) for k in names})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    titles = ["① 单个输入学取绝对值", "② 根据临时配对查答案", "③ 预测莎士比亚文本的下个字符"]
    for ax, task, title in zip(axes, ("absolute", "lookup", "language"), titles):
        for variant in VARIANTS:
            runs = [r for r in rows if r["task"] == task and r["variant"] == variant]
            if not runs:
                continue
            key = "accuracy" if task == "lookup" else ("bits_per_character" if task == "language" else "loss")
            steps = [h["step"] for h in runs[0]["history"]]
            values = np.array([[h[key] for h in r["history"]] for r in runs])
            if task == "lookup":
                values *= 100
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(steps, mean, color=COLORS[variant], label=LABELS[variant], linewidth=2.2)
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[variant], alpha=0.13)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_xlabel("训练步数")
        ax.grid(alpha=0.18)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("均方误差 ↓（对数坐标）")
    grid = np.linspace(-1, 1, 1001)
    bound = np.var(np.abs(grid))
    axes[0].axhline(bound, linestyle="--", color="#777", linewidth=1, label="最佳仿射函数的误差")
    axes[1].set_ylabel("新生成的 4096 道题：正确率 ↑")
    axes[1].axhline(12.5, linestyle="--", color="#777", linewidth=1)
    axes[1].set_ylim(0, 105)
    axes[2].set_ylabel("未参与训练的文本：bits/字符 ↓")
    fig.suptitle("没有 FFN，Attention 能学会什么？", fontsize=18, y=1.04)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.text(0.5, -0.15, "每条线为 3 个随机种子的均值，阴影为样本标准差。取绝对值任务关闭归一化；另两项保留归一化与残差。", ha="center", fontsize=10, color="#555")
    fig.tight_layout()
    fig.savefig(ROOT / "learning_curves.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant in ("standard", "attention", "memory"):
        r = next(r for r in rows if r["task"] == "absolute" and r["variant"] == variant and r["seed"] == 0)
        axes[0].plot(r["curve"]["x"], r["curve"]["prediction"], color=COLORS[variant], linewidth=2, label=LABELS[variant])
    axes[0].plot(grid, abs(grid), color="#333", linestyle="--", label="目标：|x|")
    axes[0].set_title("学到的函数（固定展示 seed 0）")
    axes[0].set_xlabel("输入 x")
    axes[0].set_ylabel("模型输出")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].bar(range(4), [summary["language"][v]["parameters"] / 1000 for v in VARIANTS], color=[COLORS[v] for v in VARIANTS])
    axes[1].set_xticks(range(4), ["标准", "Attention\n2 层", "Attention\n6 层", "Attention\n＋记忆"])
    axes[1].set_ylabel("语言模型参数量（千）")
    axes[1].set_title("去 FFN 会减少参数，因此额外比较 6 层版本")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(ROOT / "function_and_parameters.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    samples = ["固定展示：seed 0；同一提示 ROMEO:；温度 0.8；采样种子 777；各生成 240 字符。\n"]
    for variant in VARIANTS:
        r = next(r for r in rows if r["task"] == "language" and r["variant"] == variant and r["seed"] == 0)
        samples.extend(["\n===== " + LABELS[variant] + " =====\n", r["sample"] + "\n"])
    (ROOT / "samples.txt").write_text("".join(samples))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
