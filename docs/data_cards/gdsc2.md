# 数据卡：GDSC2 药敏标签

- 状态：已核实（本地实测，2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据审计 A / Team Lead
- 相关：计划书 §3.2、§3.3；`configs/data/gdsc2.yaml`；`src/kite_drp/data/labels.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据集全名 | Genomics of Drug Sensitivity in Cancer 2（GDSC2）fitted dose-response |
| 提供方 / 项目 | Wellcome Sanger Institute |
| 版本 / 发布时点 | 文件名 `27Oct23`；本地数据卡标为 8.5 代，release 号待官方复核 |
| 下载入口 | https://www.cancerrxgene.org/ |
| 下载日期 | 2026-11-13（本地归档） |
| 原始文件 | `data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv` 等 |
| 文件大小 | 35 MB（标签） |
| 存储位置 | `data/raw/gdsc/` |

## 2. 许可与引用

- 许可类型：学术使用需引用；再分发条款待核实（见 `data/licenses/gdsc2.md`）。
- 是否允许再分发：原始标签文件不提交 git；公开 release 仅保留脚本与 split。
- 必须引用：Yang et al., *Nucleic Acids Research* 2013; Iorio et al., *Cell* 2016。
- 其他限制：药物元数据/ChEMBL 派生字段可能继承 ChEMBL CC BY-SA 义务。

## 3. 内容与口径

- 规模：**242,036 行 / 969 细胞系 / 295 DRUG_ID / 286 化合物**；观测密度 84.67%（DRUG_ID 口径）。
- 标签：`LN_IC50`、`AUC`、`RMSE`、`Z_SCORE`（= 药内 z(LN_IC50)，ddof=1）、`MAX_CONC/MIN_CONC`。
- 方向：`LN_IC50` 越大越耐药；AUC 实测与 LN_IC50 正相关（r=+0.763），**越大越耐药**。
- 缺失/删失：16 列中仅 `PUTATIVE_TARGET` 缺失 11.52%；**71.81% 的 LN_IC50 外推至 MAX_CONC 之上**。
- 双 ID：9 个化合物各有两个 DRUG_ID，6,288 个重叠 (细胞系, 化合物) 对；标签差中位 1.10。

## 4. 预处理

- 脚本/配置：`src/kite_drp/data/labels.py`、`src/kite_drp/data/ids.py`、`configs/data/gdsc2.yaml`。
- 化合物键：InChIKey > 标准化药名（D23）；双 ID 主口径取 LN_IC50 均值，并保留 `delta_dual_id`。
- 细胞系键：SangerModelID → DepMap `ModelID`（ACH）；SIDM00117 用名字+COSMIC 仲裁到 ACH-002172；RH-1 无法映射，显式排除。
- 标签标准化：**不在 canonical 中 z-score**；训练时由 `FitContext` 在训练 fold 内按化合物拟合。
- 产物：`data/interim/canonical/labels.parquet`（全 969 株口径 235,748 行 / 286 化合物；映射 968 株 235,478 行；M-509 主池 126,803 行）。

## 5. 已知缺陷与风险

- `Z_SCORE` 是在全量数据上按 DRUG_ID 计算的，直接使用会造成泄漏；只能做实现校验。
- 71.81% 的 IC50 为右删失/外推，标签噪声上限高；需删失敏感性分析。
- 42 个 `CANCER_TYPE` 与 28 个 OncotreeLineage 不完全一致；组织分层以 Oncotree 为准。
- 9 个双 ID 药的两条曲线差异可达 8.98（LN_IC50），合并口径需在论文中报告。

## 6. 使用计划

- 教师/学生：主回归目标 `-z(LN_IC50)`（fold 内拟合）；AUC 仅辅助/敏感性。
- 评估：LCO（P1）、LDO（P2）、scaffold（P3，降级 target-family）、LTO（P4）。
- 外部验证：GDSC1 零样本、CTRPv2、PRISM（均只评估，不参与训练/调参）。
