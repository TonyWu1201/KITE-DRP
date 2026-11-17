# 数据卡：外部验证库（GDSC1 / CTRPv2 / PRISM）

- 状态：已预处理（2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据管线 / A
- 相关：计划书 §3.2 D12、§7.1、§10；`src/kite_drp/data/external.py`

## 1. 基本信息

| 库 | 原始文件 | 版本 | 本地位置 |
|---|---|---|---|
| GDSC1 | `GDSC1_fitted_dose_response_27Oct23.csv` | 27Oct23 | `data/raw/gdsc/` |
| CTRPv2 | `v21_auc_sensitivities.tsv` + treatment/sample | CTRPv2.1 | `data/raw/ctrp/` |
| PRISM | `secondary-screen-dose-response-curve-parameters.csv` | DepMap PRISM secondary | `data/raw/prism/` |

## 2. 许可与引用

- GDSC1：同 GDSC2，学术引用；再分发待核实。
- CTRPv2：Broad Institute；引用 Seashore-Ludlow et al. 2015 / Rees et al. 2016。
- PRISM：DepMap Repurposing；引用 Corsello et al. 2020。
- 详见 `data/licenses/{gdsc1,ctrp,prism}.md`。

## 3. 预处理

- 脚本：`kite-drp data external`（`src/kite_drp/data/external.py`）。
- 细胞系：GDSC1 用 SangerModelID/名字；CTRP 用归一化名字；PRISM 直接用 DepMap `depmap_id`。
- 药物：GDSC1 用 8.5 元数据 InChIKey/药名；CTRP/PRISM 用 SMILES→RDKit InChIKey + 药名兜底。
- 标签：
  - GDSC1：LN_IC50（敏感分数 `-LN_IC50`）；
  - CTRPv2/PRISM：AUC 原始值；实测与 GDSC2 LN_IC50 正相关（r≈0.68 / 0.57），确认 **AUC 越大越耐药**，输出 `sensitivity_score=-AUC`。
- 输出：`data/interim/external/{gdsc1_labels,ctrpv2_auc,prism_auc}.parquet`；清单 `data/splits/external_validation_manifest.json`。

## 4. 当前实测

| 库 | 行 | 细胞系 | 细胞系∩M-509 | 化合物 | ∩GDSC2 化合物 | 交集行 |
|---|---:|---:|---:|---:|---:|---:|
| GDSC1 | 314,970 | 969 | 509 | 377 | 127 | 109,636 |
| CTRPv2 | 326,148 | 832 | 420 | 84 | 84 | 59,245 |
| PRISM AUC | 701,004 | 480 | 280 | 122 | 122 | 75,488 |
| PRISM LFC | 6,360,912 | 489 | — | 1,502 treatments | 122 | 672,864 |

## 5. 已知缺陷与风险

- CTRP/PRISM 药物库与 GDSC 重叠有限（84/122 个），外验主结论应以细胞系泛化 + 药内排序为主。
- CTRP/PRISM 的 AUC 与 GDSC LN_IC50 不是同一尺度，不能直接比 RMSE。
- PRISM LFC 已转 long（6.36M 行 / 47 MB），用于固定剂量/排序分析；主评估仍建议用 dose-response AUC。
- 外部库只用于评估，不参与训练/选模/早停。

## 6. 使用计划

- GDSC1：GDSC2 训练 → GDSC1 零样本（127 个共有化合物）。
- CTRPv2/PRISM：外部细胞系泛化 + 药内排序（Spearman/PCC），报告 bootstrap CI。
