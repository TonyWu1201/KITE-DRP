# `checkpoints/` —— 正式模型权重（Git LFS）

存放**需要长期保留/发布的正式权重**（教师、学生、QAT/量化版本），走 Git LFS
（`.gitattributes` 中 `*.pt` / `*.pth` / `*.ckpt` / `*.safetensors` 等扩展名规则）。
训练过程的中间 checkpoint 留在 `experiments/<run>/`（被 git 忽略）。

## 命名约定

```
checkpoints/
├── teacher_t1_gdsc2_<YYYYMMDD>_<git-sha7>.pt
├── teacher_t0_gdsc2_<YYYYMMDD>_<git-sha7>.pt
├── student_a_gdsc2_<YYYYMMDD>_<git-sha7>.pt
├── student_a_gdsc2_<YYYYMMDD>_<git-sha7>_int8.onnx
└── <model>/<version>/...      # 版本化目录亦可
```

- 每个权重必须能对应到 `docs/model_cards/` 中的模型卡（配置、数据版本、指标、SHA256）；
- 文件名包含 git commit 前 7 位与训练日期，避免"同名不同内容"；
- 附 `.sha256` 校验文件（小文本，普通 git 跟踪亦可）。

## 当前状态

空目录（仅 `.gitkeep` 与 `.gitignore` 内规则）。训练产出权重后按上述约定归档。

## 注意

- 未安装 `git-lfs` 时 `git add` 会给出警告，此时不要提交大文件；
- 发布权重前完成 `docs/leakage_checklist.md` 与模型卡；
- 权重不得包含患者级原始数据（PHI）或可反推患者的特征。
