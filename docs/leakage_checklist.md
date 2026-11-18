# 数据泄漏审查表（Leakage Checklist）

提交论文 / 发布模型前逐条检查并填写证据（计划书 §7.1、附录 B.2-5）。
任何一条未通过都必须在结果中说明影响与缓解。

| # | 检查项 | 要求 | 状态 | 证据 / 备注 |
|---|---|---|---|---|
| 1 | 划分协议 | 主结果使用 LCO/LDO/scaffold/LTO 等非随机划分；random 仅作上限参考 | ✅ | `data/splits/p1_lco_s*.json`、`p2_ldo_s*.json`、`p3_scaffold_s*.json`、`p4_lto_s*.json`；P3 当前降级为 target-family 并在 JSON 标注 |
| 2 | 划分独立性 | 训练/验证/测试按细胞系或药物分组，同源/重复细胞系不得跨组 | ✅ | `kite-drp data validate --all` 的 split validator：group_disjoint/no_group_overlap 全绿 |
| 3 | split 文件化 | split 文件（含 group id / seed / 脚本 hash）公开且可重建 | ✅ | `data/splits/*.json` + `split_manifest.json`（sha256/config_hash/code_hash） |
| 4 | 预处理仅在训练 fold | 基因选择、归一化、HVG、通路打分参数只在训练 fold 上 fit | ✅ | `src/kite_drp/data/fit.py` 的 FitScope/Standardizer；扰动测试 4 协议全绿 |
| 5 | 补全/批次校正 fit 范围 | VAE、ComBat、相似性等只在训练集 fit，测试集 transform-only | ✅ | VAE 目标空间只由 U_vae 规则定义；`fit_stats` 逐 fold 拟合；H3CDR 相似性 `fit` 只在 train 细胞、query 只连 train 锚点（`data/similarity.py`，`tests/data/test_teacher_features.py`） |
| 6 | 测试集隔离 | 测试集不参与超参搜索、早停、模型选择 | ✅ | `train/engine.py` 早停/选模只用 val；test 仅在最终报告读取（LCO 当前未跑 test） |
| 7 | 解释超参 | 解释方法超参不在测试集上调 | ☐ | |
| 8 | TCGA 使用红线 | TCGA 不参与主模型训练/选模/早停；L3 半监督单列、不进主表 | ✅ | `TcgaEvalDataset.purpose="eval"`；TCGA 产物在 `data/interim/canonical/tcga_*`，processed 主 manifest 不包含 |
| 9 | 弱标签来源 | L2 标签来源（Ding 2016 等）与人工整理口径写清，不称"官方药敏数据" | ☐ | |
| 10 | 外部验证隔离 | 外部验证集（CTRPv2/PRISM/GDSC1）不参与训练与调参 | ☐ | |
| 11 | 数据版本 | 原始数据版本、下载日期、SHA256 可追溯（`data/raw/README.md`） | ✅ | `data/interim/audit/inventory.json`（MD5/大小/mtime/版本推断）+ canonical manifest 的 inputs |
| 12 | 随机性与种子 | 主结果 ≥3 seeds，mean±SD 报告；划分与初始化种子固定 | ☐ | |
| 13 | 负对照 | 随机/打乱教师、随机图/随机通路等负对照已跑并报告 | ☐ | §6.4 |
| 14 | 类别不平衡 | 分类指标以 AUPRC 为主并报告正类比例，FDR 校正 | ☐ | |
| 15 | 资源泄漏 | 训练时未见测试集标签/统计量；无测试集性能反馈回路 | ✅ | `tests/data/test_fit.py` 与 `kite-drp data validate` 的扰动测试：改测试 fold 标签/组学后 fit fingerprint 不变 |
| 16 | 伦理与 PHI | 仓库无 PHI；演示数据伪名化；TCGA 仅开放层 | ☐ | |

## 签署

- 检查人：
- 日期：
- git commit：
- 关联实验目录 / 模型卡：
