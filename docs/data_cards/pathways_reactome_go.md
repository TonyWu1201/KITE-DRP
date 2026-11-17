# 数据卡：Reactome / GO 通路基因集

- 状态：已生成（2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据管线 / A
- 相关：计划书 §4.2、§4.8、§5.2；D25；`scripts/download_pathways.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | Reactome（current release）+ GO Consortium（goa_human + go-basic.obo） |
| 下载时点 | 2026-11-17 |
| 文件 | `data/raw/reactome/ReactomePathways.gmt`；`data/raw/go/goa_human.gmt`、`goa_human_experimental.gmt` |
| 规模 | Reactome 2,874 通路；GO 全量 7,542 terms / 实验证据版 6,494 |
| manifest | `data/raw/pathways/download_manifest.json`（URL/sha256/日期） |

## 2. 许可与引用

- Reactome：开放许可（CC BY 4.0 量级，以官网为准），需引用 Reactome。
- GO：CC BY 4.0，需引用 GO Consortium。
- 详见 `data/licenses/reactome.md`、`data/licenses/go.md`。

## 3. 用途

- 学生表达路线 A：`pathway_score = mean z(gene)`，权重矩阵冻结为端侧资产（D25）。
- 教师可解释性：通路归因与药物机制描述。
- `U_pathway`：Reactome+GO 基因并集 ∩ `U_teach_ge`。

## 4. 预处理

- Reactome GMT 直接使用 gene symbol。
- GO GAF（human，排除 NOT）聚合为 GMT；`experimental` 版排除 IEA/ND，默认不进主表。
- 过滤：每通路 ≥10 个 `U_teach_ge` 基因、≤500 基因；按来源各取 top 500（共 1,000）控制端侧维度。
- 产物：`data/interim/canonical/pathways.parquet`；`data/processed/student_inputs/pathway_weights.npz`。

## 5. 已知缺陷与风险

- GO 注释含 IEA 电子推断；主表用全量，实验证据版仅作消融。
- 通路基因集会随 release 更新；必须记录 URL + sha256 + 下载日期。
- 1,000 条上限是端侧预算，不是生物学上最优；论文需说明选择规则。
- KEGG 不在本数据卡内（不可再分发）。

## 6. 使用计划

- 学生路线 A（默认）；路线 B（HVG）保留作对照。
- 教师解释与 E2/E3 消融。
