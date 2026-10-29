# 实验来源与发布记录

- 实验时间：2026-10-28（Asia/Shanghai）。
- 整理公开：2026-10-29。
- 定位：个人学习与好奇心实验；不是新方法论文，不是下述论文的完整复现。
- 代码与实验由仓库所有者在 AI 编程助手协助下完成。

## 原始训练档案

`experiment.py` 保持实际训练时的字节内容；其 SHA-256 由训练前写入的 `results/protocol.json` 记录。33 个结果 JSON 与对应 `.pt` 文件均从原始实验复制，未重新训练、选择最好 checkpoint 或更换分数。

原始实验的 81 个文件校验均在发布整理前通过。公开仓库额外增加了独立安装说明、依赖清单、校验脚本与 CI；绘图脚本仅把固定的系统字体路径改为字体查找。图表、汇总和生成样例沿用原始实验。仓库 `SHA256SUMS` 对应整理后的公开档案，不应当拿它代替训练协议中保存的代码哈希。

临时日志、短跑检查输出、Python 缓存、环境目录以及同目录中的其他项目不在本仓库内。

## 数据

数据为 [karpathy/char-rnn 的 Tiny Shakespeare 文本](https://github.com/karpathy/char-rnn/blob/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt)，保留原始文本供离线重算。文本内容来自莎士比亚作品。

- 上游文件最近修改提交：`370cbcd448eb7daf32f21a6be560b70e0b33c4e3`，2015-05-21。
- Git blob SHA-1：`7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`。
- 文件 SHA-256：`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`。
- 文件大小：1,115,394 字节。

公开前已独立核对 GitHub 上游文件元数据与本地 Git blob 哈希一致。字符词表、90%/10% 切分及精确样本数在训练协议中。

## 方法参考

- Sukhbaatar et al., [Augmenting Self-attention with Persistent Memory](https://arxiv.org/abs/1907.01470)：可学习记忆版本的机制参考，没有使用其代码或完整实验配置。
- Dong et al., [Attention is not all you need: pure attention loses rank doubly exponentially with depth](https://proceedings.mlr.press/v139/dong21a.html)：用于理解特定纯注意力结构的限制；本实验没有复现其秩坍缩理论或实验。

## 原始指标与验证

`results/` 保留全部随机种子，`summary.json` 和 `runs.csv` 保留汇总与单次指标，`diagnostics.json` 标明训练后追加计算的简单频率参照。这些参照没有用于调参或挑选模型。

`verify.py` 检查公开文件哈希、训练代码/数据来源、33 个模型与结果的一一对应、参数量和模型结构。可选 `--reload` 会在固定评估集上重算分数。CPU 与不同硬件/版本允许损失有轻微浮点差异，不承诺跨平台逐位相同。
