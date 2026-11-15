# `experiments/` —— 实验输出目录（不提交 git）

每次训练/评估/导出实验一个子目录，由 Hydra 或实验脚本自动创建；整个目录被
`.gitignore` 忽略，仅保留本 README 与 `.gitkeep` 以维持目录结构。

## 目录约定

```
experiments/
└── <YYYY-MM-DD_HH-MM-SS>_<job_name>/
    ├── config.yaml        # 完整配置快照（Hydra 自动）
    ├── overrides.yaml     # 命令行覆盖项
    ├── git_commit.txt     # git commit / 分支 / 是否 dirty
    ├── metrics.json       # 最终指标
    ├── metrics.csv        # 逐 epoch/逐 fold 指标
    ├── logs/              # stdout/stderr 日志
    ├── predictions/       # 测试集预测（可选，注意不含 PHI）
    └── checkpoints/       # 训练中间权重（不提交；正式权重另存 checkpoints/）
```

## 实验台账（Experiment Ledger）

每次正式实验在下面追加一行（计划书 §7.6：日期/假设/config hash/结果/结论/下一步）。
完整台账也可拆分为 `docs/` 下独立表格；本文件保留**最近实验**速查。

| 日期 | 假设 | config hash / 目录 | 主要结果 | 结论 | 下一步 |
|---|---|---|---|---|---|
| — | — | — | — | — | 尚未开始实验（G0 前） |

## 可复现三件套

`git commit + config 快照 + seed` 缺一不可；主结果至少 3 seeds（计划书 §7.1）。
实验目录名与 `config.yaml` 必须能对应回当时的代码版本。

## 大文件

- 实验中间权重留在本目录（被忽略，不入库）；
- 需要发布/复用的正式权重放 `checkpoints/`（Git LFS 跟踪）；
- 预测结果等大文件如确需提交，走 `data/processed/` 或 LFS 规则，不要直接提交到 experiments/。
