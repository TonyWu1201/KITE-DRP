# 数据卡：GDSC 药物结构（SMILES，PubChem）

- 状态：已生成（2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据管线 / B
- 相关：计划书 §3.3-6、§4.4；D25；`scripts/download_smiles_pubchem.py`；`src/kite_drp/data/drugs.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | PubChem PUG-REST（经 `pubchempy`） |
| 版本/时点 | 2026-11-17 抓取 |
| 输入 | GDSC2 286 化合物（`compounds.parquet`） |
| 文件 | `data/raw/drugs/smiles.tsv`、`smiles_manifest.json`、`smiles.parquet` |
| 覆盖率 | **286/286 = 100%**；233 个按 InChIKey、53 个按 PubChem CID |
| 失败 | 0 |

## 2. 许可与引用

- 许可：PubChem 公共数据；再分发/商用前核对 NCBI PubChem 条款。
- 署名：引用 PubChem 与 `pubchempy`。
- 详见 `data/licenses/pubchem.md`。

## 3. 内容与口径

- 每个化合物保存 isomeric SMILES（缺失时回退 canonical SMILES）。
- 数字药名按 PubChem CID 查询（53 个），其余按 InChIKey；无 InChIKey 时按名称/同义词。
- 保留 `pubchem_cid`、`pubchem_inchikey`、`inchikey_match`、`source`、`query`，便于审计。
- 盐型/立体化学由 PubChem 记录决定；不保证与 GDSC 实验所用盐型一致。

## 4. 预处理

- 脚本：`scripts/download_smiles_pubchem.py`（可续传、限速、失败记录）。
- 安装：`uv sync --extra data`（pubchempy）；Morgan FP/scaffold 需要 `rdkit`（`graph` extra）。
- 产物：`data/raw/drugs/smiles.tsv` → `build_drug_features` → 286 × 2048 Morgan FP、277 个 Bemis–Murcko scaffold。

## 5. 已知缺陷与风险

- 9 个无环/无机化合物没有 Murcko scaffold（如 Cisplatin、N-acetyl cysteine、glutathione），P3 中作为 singleton 组。
- PubChem 名称匹配可能返回盐型/异构体；`pubchem_inchikey` 可作为二次校验。
- PubChem 更新/下架可能导致重跑结果变化；`smiles_manifest.json` 记录抓取时点。
- SMILES 派生特征（FP/scaffold）可再分发性需与 PubChem 条款一致。

## 6. 使用计划

- 教师药物图 GIN、Morgan FP、Bemis–Murcko scaffold。
- P3 scaffold split：scaffold 相同的化合物不得跨 train/test。
- 学生 B 档药物特征。
