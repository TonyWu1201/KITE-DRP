# 数据卡（Data Cards）

每个数据集一张数据卡，记录**来源 / 版本 / 许可 / 预处理脚本 / 已知缺陷**（计划书 §3.8）。
数据卡随仓库跟踪（文本），原始文件本身不提交（`data/raw/` 被忽略）。

## 清单与状态

| 数据卡 | 数据集 | 角色 | 状态 |
|---|---|---|---|
| `gdsc2.md` | GDSC2 药敏标签 | 主训练标签（§3.2 D11） | TODO（已有原始文件与 `data/raw/gdsc/README.md`，待正式数据卡） |
| `depmap.md` | CCLE/DepMap 组学（GE/MU/CNV，METH 见 ccle） | 教师组学 + 学生输入 | TODO |
| `ccle.md` | 旧 CCLE 甲基化/注释（RRBS，非 450K） | 教师 METH（可选） | TODO（注意版本旧、需数据卡注明） |
| `string.md` | STRING v12.5 PPI | 教师图先验（CC BY 4.0） | TODO |
| `dgidb.md` | DGIdb v5.0.13 药物-基因交互 | 解释校验/兜底先验 | TODO |
| `hgnc.md` | HGNC 基因主表 | 基因 ID 映射 | TODO |
| `tcga-cesc.md` | TCGA-CESC 四模态原始下载 | L1/L2/L2'/L3（§3.7、§8） | TODO（已有 raw README 摘要） |
| `tcga-cdr.md` | TCGA-CDR 生存终点 | L2' 生存一致性 | TODO |
| `ding2016.md` | Ding 2016 响应标签补充材料 | L2 弱标签唯一来源 | TODO（标签来源与口径必须写清） |
| 待补 | CTRPv2 / PRISM / GDSC1 | 外部验证（D12） | 未下载 |

> 原始数据的文件清单、大小、版本与用途摘要见 `data/raw/README.md`；
> 许可说明见 `data/licenses/`。

## 模板

见 [`_TEMPLATE.md`](_TEMPLATE.md)。新增数据卡时复制模板，文件名用 `<dataset>.md`。

## 红线

- 不把受限/受控数据（TCGA 受控层、DrugBank 等）打包进公开 release（§附录 B.2-8）；
- 数据卡必须写清**已知缺陷**（如 PRISM 非 IC50、MKDR 标签来源不明、CCLE 甲基化平台差异）；
- 任何"再整理"的标签（如 Ding 2016）必须注明原始来源与人工整理口径，不得称为官方药敏数据。
