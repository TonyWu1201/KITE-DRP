# 架构决策记录（ADR）

每个重要工程决策一份 ADR，记录**背景 → 选项 → 决策 → 后果**，避免同样的讨论重复发生。
主计划书的「未定项决策日志」（§12 D1–D19）是学术/方案层面的决策索引；本目录记录
**工程实现层面**的决策。

## 模板

新建文件 `docs/adr/NNNN-<kebab-case-title>.md`，内容结构：

```markdown
# ADR-NNNN：标题

- 状态：提议 / 已接受 / 已废弃 / 被 ADR-XXXX 取代
- 日期：YYYY-MM-DD
- 相关：计划书章节 / issue / PR

## 背景
为什么需要做这个决策？约束是什么？

## 选项
1. 选项 A：优缺点
2. 选项 B：优缺点

## 决策
选择了哪个，为什么。

## 后果
正面影响、负面影响、迁移成本、后续动作。
```

## 索引

| 编号 | 标题 | 状态 | 日期 |
|---|---|---|---|
| [0001](0001-repository-structure-and-data-flow.md) | 仓库结构与数据分层重构 | 已接受 | 2026-11-15 |
| [0002](0002-python-environment-cuda126.md) | Python 环境：uv + 双平台 + CUDA 12.6 | 已接受 | 2026-11-15 |
| [0003](0003-deploy-server-and-submodule.md) | 端侧部署服务器与子模块策略 | 已接受 | 2026-11-15 |
| [0004](0004-data-pipeline-g0-decisions.md) | G0 数据管线实现口径与两处实测偏差 | 已接受 | 2026-11-17 |
| [0005](0005-processed-self-contained-teacher-inputs.md) | processed 自包含（教师训练可脱离 raw/canonical） | 已接受 | 2026-11-18 |
| [0006](0006-dti-source-open-targets.md) | DTI 主源切换为 Open Targets，停用 DGIdb | 已接受 | 2026-11-20 |
