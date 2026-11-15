# 树莓派边缘端部署脚手架（`deploy/edge/`）

本目录描述 KITE-DRP 在**树莓派（Raspberry Pi 5 主目标 / Pi 4B 兼容下限）**上的安装、
服务化与离线部署包校验方式。一期只提供**文档 + 最小可复用脚本**；真实装置上的完整
系统化部署（签名链、双目录原子替换、健康监控）见 [`deploy/device/README.md`](../device/README.md)。

> ⚠️ 科研原型，不用于临床决策。硬件选型、延迟预算与量化策略依据总计划书 §9.1–§9.3。

---

## 1. 硬件与软件基线（计划书 §9.1–§9.2）

| 项目 | Raspberry Pi 5（8 GB，主） | Raspberry Pi 4B（4 GB，兼容下限） |
|---|---|---|
| 推理栈 | ONNX Runtime CPU EP（aarch64 wheel） | 同左 |
| 量化 | INT8（动态，实测为准） | **默认 FP32**（Cortex-A72 无 SDOT，INT8 可能更慢，计划书 §9.2 #4） |
| 端到端延迟目标 | p50 ≤100 ms、p95 ≤200 ms | p50 ≤300 ms、p95 ≤500 ms |
| 峰值内存 | 进程 ≤300 MB | 同左 |
| 功耗/温度 | 整机 ≤9 W；SoC ≤75℃（主动散热） | 视散热条件 |

**装置侧不安装 PyTorch**：训练在 x86 完成，装置只消费 ONNX（省 ~1 GB 依赖）。
服务端（`deploy/server`）本身只依赖 `fastapi/pydantic/uvicorn` + 可选 `onnxruntime`，
aarch64 下用离线 wheelhouse 安装（官方提供 aarch64 manylinux wheel）。

> Python 版本要求：`deploy/server` 要求 **Python ≥ 3.12**。Pi OS Bookworm 自带 3.11，
> 需使用 **Pi OS Trixie（3.13）** 或随包携带 3.12 运行时（`uv python install` 离线包），
> 否则只能把本服务跑在 x86 医生站主机、Pi 侧仅承担后续的 ONNX CLI 推理（二期）。

---

## 2. 离线 bundle 约定（训练机打包 → U 盘进院）

```text
kite-drp-offline-<version>/
├── server/                  # deploy/server 源码（pyproject.toml + uv.lock + app/ + static/）
├── packages/
│   └── current/             # 部署包：manifest.json + model.onnx（计划书 §9.2/§9.4）
│       ├── manifest.json
│       ├── model.onnx
│       └── SHA256SUMS       # 可选：整包清单（signature 方案见 device/README.md）
├── wheelhouse/              # 离线依赖 wheel（cp312 + manylinux aarch64）
│   ├── fastapi-*.whl
│   ├── pydantic-*.whl
│   ├── uvicorn-*.whl
│   ├── onnxruntime-*.whl
│   └── ...
└── SHA256SUMS / 签名文件    # 整包校验与签名（部署前必需）
```

wheelhouse 生成（**在联网的 x86 训练机上**，目标平台 aarch64）：

```bash
# 示例：用 uv 下载 aarch64/cp312 的离线 wheel（版本以 deploy/server/uv.lock 为准）
mkdir -p deploy/packages/wheelhouse
uv pip download --only-binary :all: --python-platform aarch64-unknown-linux-gnu \
  --python-version 3.12 --dest deploy/packages/wheelhouse \
  fastapi pydantic uvicorn onnxruntime
```

> 输出目录同样位于 `deploy/packages/**`（Git LFS）；随包里不要包含受控数据（§3.8）。

---

## 3. 安装步骤（树莓派，离线）

```bash
# 1) 从 U 盘拿到 bundle，校验整包签名/校验和（见 device/README.md 的签名方案）
sha256sum -c kite-drp-offline-<version>/SHA256SUMS

# 2) 离线安装（脚本会先校验 manifest + model SHA256，再建 venv 装依赖）
cd kite-drp-offline-<version>
./server/../scripts/install_offline.sh --bundle "$PWD"
# 等价：deploy/edge/scripts/install_offline.sh --bundle "$PWD" --target ~/.local/share/kite-drp/server
```

脚本（`scripts/install_offline.sh`）做的事：

1. 检查 `python3 >= 3.12`；
2. 调用 `scripts/verify_package.sh` 校验部署包（manifest 必填字段 + **model SHA256**）；
3. 把 `server/` 源码复制到安装目录（保留旧目录以便回滚）；
4. 离线安装依赖：优先 `uv sync --frozen --offline --no-index --find-links wheelhouse`，
   否则 `python3 -m venv` + `pip --no-index --find-links`；
5. 打印启动命令与下一步（systemd 见 device/README）。

手动启动（等价）：

```bash
cd ~/.local/share/kite-drp/server
KITE_MODEL_DIR=~/kite-drp-offline-<version>/packages/current \
  .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health       # status=ok, demo=false, model_version=<版本>
```

装置浏览器访问 `http://127.0.0.1:8000`（同机浏览器/触屏终端；不开放到医院网）。

---

## 4. 部署包校验脚本（`scripts/verify_package.sh`）

可独立于服务使用（训练机/装置/验收都适用）：

```bash
deploy/edge/scripts/verify_package.sh deploy/packages/current
# [OK] 部署包校验通过
#      model_version  = 0.1.0
#      model_file     = model.onnx (1.23 MiB)
#      model_sha256   = <64 hex>
#      mutation_genes = 523
#      expression_genes = 523
#      drug_index     = 198
```

校验规则（与 `deploy/server` 启动时一致，计划书 §9.3 #10）：

- `manifest.json` 存在/可解析，含 `model_version`、`model_sha256`、`outputs.response`；
- `model.onnx`（或 `manifest.model_file`）存在；
- 实际 SHA256 与 manifest **完全一致**，否则拒绝部署；
- 退出码：0=通过，1=校验失败，2=用法/文件缺失（便于 CI/验收脚本判断）。

---

## 5. 服务化（systemd）与设备侧运维

开机自启、systemd unit、双目录原子替换、回滚、签名校验的完整方案在
[`deploy/device/README.md`](../device/README.md)。最小参考：

```ini
# /etc/systemd/system/kite-drp-server.service（节选，完整版见 device/README.md）
[Service]
User=kite
WorkingDirectory=/opt/kite-drp/current
Environment=KITE_MODEL_DIR=/opt/kite-drp/current/packages/current
Environment=KITE_SERVER_HOST=127.0.0.1
ExecStart=/opt/kite-drp/current/server/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
```

---

## 6. 性能验收（与计划书 §9.3 对应）

在装置上接到真实部署包后，至少验证（`deploy/server` 的 `/health` 与任务 `elapsed_ms` 可提供初步数据）：

| 验收项 | 方法 | 目标 |
|---|---|---|
| #1 纯推理延迟 | 连续提交 ≥100 次单样本任务，读 `result.elapsed_ms` p50/p95 | Pi5 p50 ≤20 ms；Pi4 p50 ≤60 ms |
| #2 端到端延迟 | `curl` 提交→轮询终态总耗时（排除排队） | Pi5 p50 ≤100 ms；Pi4 p50 ≤300 ms |
| #3 冷启动 | 服务启动到 `/health` 首次可用的时间 | Pi5 ≤5 s；Pi4 ≤10 s |
| #4 峰值内存 | `systemd-cgtop` / `ps -o rss` 观察 | 进程 ≤300 MB |
| #10 离线合规 | 装置断网运行；`ss -tnp` 无外部连接；前端无外链 | 运行期 0 次网络请求 |
| #11 更新/回滚 | 双目录原子替换 + 回滚演练（device/README.md） | 升级后 ≤2 分钟回滚 |
| #7/#8 确定性/稳定性 | 同一输入重复推理逐位一致（INT8 QDQ）；连续 1000 次 0 失败 | 见计划书 §9.3 |

> INT8 与 FP32 精度对照（ΔAUROC/ΔAUPRC 等）在训练机侧 GDSC/CTRPv2 测试集完成
> （计划书 §9.3 #6），端侧只做一致性/延迟验证。

---

## 7. 与其它目录的关系

| 目录 | 关系 |
|---|---|
| `src/kite_drp/deploy/` | 训练机侧导出 ONNX/量化/打包（生成 `deploy/packages/` 内容） |
| `deploy/server/` | 本机 Web 服务源码（树莓派与 x86 医生站共用同一份代码） |
| `deploy/packages/` | 部署产物（ONNX + manifest + wheelhouse；Git LFS） |
| `deploy/device/` | systemd/原子替换/回滚/签名校验（设备侧运维） |

---

## 8. 一期 / 二期

- **一期（本任务）**：离线安装与部署包校验脚本、服务化与设备运维**文档化**、服务端可跑通；
- **二期**：`deploy/server` 拆独立仓库/子模块后的端侧 CLI（`kite-drp-edge predict`）、
  签名链自动化、A/B 分区或双目录自动切换、设备健康监控与远程（院内网）升级管理。

## 免责声明

本目录为科研原型部署脚手架，不用于临床决策；所有装置部署须遵守《个人信息保护法》
《人类遗传资源管理条例》与医院伦理/信息安全要求（总计划书 §9.4）。
