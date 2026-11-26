# 实验 8：RNN / LSTM / Transformer / BERT 情感分类对比

数据为 ChnSentiCorp 中文酒店评论二分类。模板要求：用现有 tokenizer 从零训练 RNN、LSTM、Transformer Encoder，再微调预训练 BERT，至少对比 3 种结果。

## 数据口径

| 项目 | 数值 |
|---|---|
| train 有效行 | 9146 |
| train 去重后 | 7904（22 条同文本不同标签被剔除） |
| dev 去泄漏 | 876 |
| test 全量 / 去泄漏 | 1200 / 851 |
| train-test 重复文本 | 314（占 test 唯一文本 26.7%） |

主表同时报 `test_all` 与 `test_clean` 两个口径；dev 只用于早停，test 只在最后评一次。

## 目录

```text
exp8/
  chnsenticorp/          # 原始 tsv
  src/                   # 数据、模型、训练、评测、绘图
  scripts/               # 云端一键脚本
  outputs/
    runs/<run>/          # config/metrics/predictions/env
    figures/             # fig1~fig6
    summary.csv          # 单次运行汇总
    summary_grouped.csv  # 按模型跨 seed 的 mean/std
  README-cloud.md        # 优云智算 4090 操作步骤
  requirements-cloud.txt # 云端依赖（不含 torch）
```

## 本地（CPU）冒烟

```bash
cd exp8
python -m src.eda
python -m src.toys
python -m src.train_scratch --model lstm --smoke --device cpu --amp off
python -m src.train_bert --model hfl/rbt3 --smoke --device cpu --amp off
```

## 本地小规模调试

```bash
python -m src.train_scratch --model rnn --epochs 1 --max-len 64 --output-dir outputs_debug --device cpu --amp off
python -m src.summarize --output-dir outputs_debug
python -m src.make_figures --output-dir outputs_debug
```

## 云端全量

见 `README-cloud.md`。云端运行结束后把 `outputs_*.tar.gz` 解压回本目录，再执行：

```bash
python -m src.summarize
python -m src.make_figures
```

## 实验设计

- 从零模型统一：BERT 词表 + 随机 embedding（128 维），最大长度 192，batch 256，AdamW，dev 早停。
- RNN / LSTM：单层，隐藏 128，掩码平均池化。
- Transformer：2 层 Encoder，4 头，d_model 128，FFN 256，学习式位置编码，掩码平均池化。
- 预训练微调：`bert-base-chinese` 与 `hfl/rbt3`，batch 64，lr 2e-5，3 epoch，线性 warmup 10%。
- 去泄漏评测：排除与 train/dev 文本完全相同的 test 行。
- 4090 上 bf16 + TF32，3 个 seed（42/43/44）报 mean±std。