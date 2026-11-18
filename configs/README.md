# `configs/` 实验配置（Hydra / OmegaConf）

所有实验由配置驱动，禁止在代码里硬编码路径与超参（计划书附录 B.2-1）。
配置采用 Hydra 的组合式结构：`config.yaml` 是根配置，通过 `defaults` 组合各配置组；
运行时用 `--config-name config key=value` 覆盖。

## 配置组

| 组 | 目录 | 说明 | 计划书 |
|---|---|---|---|
| `paths` | `paths/` | 仓库内相对路径（唯一路径来源） | §3.8、附录 B |
| `data` | `data/` | 数据集、标签方向、教师/学生模态集合、组学/先验口径 | §3、§5.1 |
| `split` | `split/` | P0–P4/LPO 划分协议 | §7.2 |
| `teacher` | `teacher/` | B0/T0/T1 教师结构与规模 | §4 |
| `similarity` | `similarity/` | H3CDR 式细胞相似性（fold-safe） | §4.4、E-sim |
| `train` | `train/` | 优化器/早停/AMP 训练口径 | §6.3 |
| `student` | `student/` | A/B/S-latent 学生结构与参数预算 | §5.4–5.5 |
| `kd` | `kd/` | 四层蒸馏信号与权重（解释蒸馏默认关闭） | §6.2 |
| `edge` | `edge/` | 端侧平台、量化与验收阈值 | §9 |

数据管线入口（不依赖 Hydra，可直接运行）：

```bash
.venv/bin/python -m kite_drp data canonical --config configs/data/gdsc2.yaml
.venv/bin/python -m kite_drp data splits --all --seeds 0,1,2
.venv/bin/python -m kite_drp data fit-stats
```

`configs/data/student.yaml` 与 `configs/data/tcga.yaml` 是数据侧子配置（学生输入/TCGA 评估），
当前由 `data.cli` 的默认值与 `configs/data/gdsc2.yaml` 合并使用。

## 约定

1. **路径**：一律写相对仓库根目录的路径，由 `kite_drp.common.paths` 解析；
   可通过环境变量 `KITE_DRP_ROOT` 把仓库映射到其他位置（容器/服务器）。
2. **组合示例**：

   ```bash
   # 教师 T0 + LCO 划分
   uv run python -m kite_drp.train.teacher_train --config-name config teacher=t0 split=lco
   # 学生 A 档 + 输出软标签蒸馏
   uv run python -m kite_drp.train.student_train --config-name config kd=soft_only
   ```

3. **实验目录**：Hydra 输出写入 `experiments/<时间戳>_<job>/`，该目录被 git 忽略
   （仅保留 `experiments/README.md` 与 `.gitkeep`）；正式权重放 `checkpoints/` 走 Git LFS。
4. **本地覆盖**：机器相关的临时覆盖写 `configs/local/*.yaml`，整个目录被 git 忽略。
5. **可复现三件套**：`git commit + 配置快照 + seed`，主结果 ≥3 seeds（§7.1）。
