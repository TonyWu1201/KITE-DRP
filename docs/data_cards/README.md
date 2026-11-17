# 数据卡（Data Cards）

每个数据集一张数据卡，记录**来源 / 版本 / 许可 / 预处理脚本 / 已知缺陷**（计划书 §3.8）。
数据卡随仓库跟踪（文本），原始文件本身不提交（`data/raw/` 被忽略）。

## 清单与状态

| 数据卡 | 数据集 | 角色 | 状态 |
|---|---|---|---|
| `gdsc2.md` | GDSC2 药敏标签 | 主训练标签（§3.2 D11） | 已核实（2026-11-17 实测） |
| `depmap_omics.md` | DepMap Model/GE/MU/CNV | 教师组学 + 学生输入 | 已核实（版本以 MD5 冻结） |
| `ccle_methylation.md` | 旧 CCLE 甲基化（RRBS TSS1kb） | 教师 METH（可选+掩码） | 已核实，默认关闭（D21） |
| `priors_string_hgnc_dgidb.md` | STRING / HGNC / DGIdb | 教师图先验 + 基因真源 | 已核实（STRING CC BY 4.0） |
| `tcga_cesc.md` | TCGA-CESC 四模态 | L1/L2/L2′/L3（评估域） | 离线部分已核实；file→case 映射待建表（D29） |
| `ding2016.md` | Ding 2016 响应标签 | L2 弱标签唯一来源 | 已核实；仅 Cisplatin-CESC n=59 |
| 待补 | CTRPv2 / PRISM / GDSC1 | 外部验证（D12） | 未下载 |

> 原始数据的文件清单、大小、版本与用途摘要见 `data/raw/README.md`；
> 许可说明见 `data/licenses/`。自动质量报告见 `data/interim/audit/quality_report.md`（不进 git）。

## 模板

见 [`_TEMPLATE.md`](_TEMPLATE.md)。新增数据卡时复制模板，文件名用 `<dataset>.md`。

## 红线

- 不把受限/受控数据（TCGA 受控层、DrugBank 等）打包进公开 release（§附录 B.2-8）；
- 数据卡必须写清**已知缺陷**（如 PRISM 非 IC50、CCLE 甲基化平台差异、DepMap 版本混用）；
- 任何"再整理"的标签（如 Ding 2016）必须注明原始来源与人工整理口径，不得称为官方药敏数据。
