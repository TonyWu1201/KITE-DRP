# 数据卡：CCLE 甲基化（RRBS TSS1kb，可选模态）

- 状态：已核实（本地实测，2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据审计 B
- 相关：计划书 §3.3-4、§5.3；决策 D21；`src/kite_drp/data/omics/methylation.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据集 | CCLE DNA methylation TSS1kb |
| 提供方 | Broad/CCLE 旧资源（2018） |
| 版本/时点 | 文件 mtime 2018-12-27；hg19 |
| 文件 | `data/raw/ccle/CCLE_DNA_methylation_TSS1kb.txt` |
| 大小 | 142 MB |
| 规模 | 21,338 行（1 行占位）× 843 细胞系；17,181 基因（窗口聚合前） |

## 2. 许可与引用

- 许可：CCLE 旧资源条款待核实（`data/licenses/depmap.md` 同源部分）。
- 再分发：原始文件与完整派生矩阵不进入公开 release。
- 引用：CCLE/DepMap 相关论文；具体 RRBS 出处待核实。

## 3. 内容与口径

- **平台真相：RRBS（reduced representation bisulfite sequencing）TSS±1 kb 汇总，不是 450K beta**。
- 行名 `GENE_chr_start_end`，窗口恒 1,000 bp；`avg_coverage` 为窗口级平均深度（中位 194.95）。
- 值域 [0,1]；27.1% 恰 0、3.3% 恰 1；总缺失 5.47%，**MNAR（随 coverage 下降）**。
- 843 列中 10 列全空；833 可用；∩GDSC2 559 株，∩主池 404 株。
- 旧符号：95.2% 可映射当前 HGNC；13,691 个 protein-coding 基因。

## 4. 预处理（默认关闭）

- 删 `NA_NA` 行与 10 个全空列 → 多 TSS 窗口按基因取均值 → HGNC 唯一映射 → `logit(clip(1e-3))`。
- 缺失保留 mask，不做均值插补。
- 产物（仅启用时）：`data/interim/canonical/methylation_rrbs*.parquet`。

## 5. 已知缺陷与风险

- 跨平台：TCGA 为 HM450，CCLE 为 RRBS；**VAE METH 头与 TCGA 域迁移默认不做**（D21）。
- 缺失机制为 MNAR，不能按随机缺失处理。
- 覆盖细胞系少（主池 404），加入主表会显著缩小训练集。
- 2018 旧符号/hg19，与当前 DepMap hg38 不一致。

## 6. 使用计划

- 主教师：**不使用**；仅 E2-METH 消融（可选+掩码）。
- 学生：不输入 METH。
- TCGA L1：不把 METH 作为补全目标。
