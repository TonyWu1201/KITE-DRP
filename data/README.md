# `data/` —— 数据分层与版本管理

遵循计划书 §3.8：原始数据与派生产物**分层存放**，通过 `.gitignore` / `.gitattributes`
控制哪些进 git、哪些进 Git LFS、哪些完全不进仓库。

```
data/
├── raw/            # 下载得到的原始文件（不提交 git；仅保留 README/CHECKSUMS/.gitkeep）
├── interim/        # 中间产物：基因对齐、去重、归一化（不提交 git）
├── processed/      # 模型输入缓存，版本化（Git LFS 跟踪）
├── splits/         # split_*.json（Git LFS 跟踪；含 group id / seed / 脚本 hash）
└── licenses/       # 每个数据源的许可说明与引用要求（普通 git 跟踪）
```

## 当前状态（2026-11-15）

- `data/raw/`：已下载 GDSC2、DepMap（Model/表达/突变/CNV）、旧 CCLE、HGNC、STRING v12.5/v12.0、
  Open Targets 26.09（DTI 主源）、TCGA-CESC 四模态（GDC Release 46.0，1,204 文件 6.421 GB）
  + 临床/CDR/Ding2016/manifests；DGIdb v5.0.13 已停用，仅留档。
  总大小约 **8.3 GB**。完整清单与大小见 [`raw/README.md`](raw/README.md)、
  非 TCGA 文件校验和见 [`raw/CHECKSUMS.md5`](raw/CHECKSUMS.md5)。
- `data/interim/`、`data/processed/`、`data/splits/`：尚未生成（G0 门禁前完成管线）。
- `data/licenses/`：已有数据源许可摘要，正式逐源许可文件待补（见该目录 README）。

## 版本管理规则

| 目录 | 是否进 git | 机制 |
|---|---|---|
| `data/raw/` | 否（仅 README/CHECKSUMS/.gitkeep） | `.gitignore` |
| `data/interim/` | 否（仅 `.gitkeep`） | `.gitignore` |
| `data/processed/` | 是 | Git LFS（`.gitattributes`） |
| `data/splits/` | 是 | Git LFS（`.gitattributes`） |
| `data/licenses/` | 是（文本） | 普通 git |

## Git LFS 使用

```bash
# 每台机器执行一次（需先安装 git-lfs）
git lfs install

# 拉取 LFS 文件（clone 后或切换分支后）
git lfs pull

# 查看当前跟踪的文件
git lfs ls-files

# 检查某个文件的属性（应输出 filter: lfs）
git check-attr filter -- data/processed/example.parquet
```

新增需要 LFS 跟踪的**目录**时，在 `.gitattributes` 加一行 `path/** filter=lfs diff=lfs merge=lfs -text`，
并保留目录内 `.gitkeep`（`.gitattributes` 末尾已把 `**/.gitkeep` 排除在 LFS 之外）。

## 数据卡与许可

- 数据卡：`docs/data_cards/`（来源/版本/许可/预处理/已知缺陷）
- 许可说明：`data/licenses/`
- 红线：受控数据（TCGA 受控层、需授权 DrugBank 等）不提交公开仓库；公开 release 只含
  下载脚本、预处理代码、split 文件与派生特征（计划书附录 B.2-8）。
