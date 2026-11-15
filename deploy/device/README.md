# 设备侧运维规划（`deploy/device/`）— 一期文档化

本文件规划 KITE-DRP 端侧装置（树莓派 / 医生站迷你主机）的**服务化、开机自启、
双目录原子替换、一键回滚、签名校验**。一期只文档化 + 给出可直接采用的配置模板；
自动化安装/升级脚本放到二期（与 `deploy/edge/scripts/install_offline.sh` 衔接）。

对应验收项：计划书 §9.3 #10（离线合规）、#11（更新/回滚：**签名验签失败拒绝加载、
升级后 ≤2 分钟回滚**）、#4（内存 ≤300 MB）、#9（功耗/温度）。

> ⚠️ 科研原型装置，不用于临床决策。

---

## 1. 目录布局（推荐）

```text
/opt/kite-drp/
├── releases/
│   ├── 0.1.0/                 # 每个版本一个只读目录
│   │   ├── server/            # deploy/server 源码 + .venv
│   │   └── packages/current/  # 该版本部署包（manifest.json + model.onnx）
│   └── 0.2.0/
├── current -> releases/0.2.0  # 原子软链：切换版本只改这里
└── previous -> releases/0.1.0 # 保留上一版用于快速回滚（或保留最近 N 个 releases）

/etc/kite-drp/
└── kite-drp.env               # 环境变量（KITE_*，权限 0640，root:kite）

/var/lib/kite-drp/             # 运行时数据（一期服务为纯内存队列，实际只放日志/标记）
```

- `releases/<version>` **只读**（`chmod -R a-w`，root 所有），服务以普通用户 `kite` 运行；
- 版本切换 = **原子替换 `current` 软链**（`ln -sfn` 到临时名再 `mv -T`），无需搬文件；
- 服务启动时校验部署包 SHA256（见 `deploy/server` §6），校验失败即拒绝加载真实模型。

---

## 2. systemd unit（开机自启）

`/etc/systemd/system/kite-drp-server.service`：

```ini
[Unit]
Description=KITE-DRP 本机推理服务（科研原型，不用于临床决策）
Documentation=file:/opt/kite-drp/current/server/README.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=kite
Group=kite
WorkingDirectory=/opt/kite-drp/current/server
EnvironmentFile=/etc/kite-drp/kite-drp.env
Environment=KITE_SERVER_HOST=127.0.0.1
Environment=KITE_MODEL_DIR=/opt/kite-drp/current/packages/current
Environment=KITE_LOG_LEVEL=INFO
ExecStart=/opt/kite-drp/current/server/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=3
# 计划书 §9.3 #4：进程 RSS ≤300MB
MemoryMax=300M
# 安全加固：无网络出站、无新特权、只读系统
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/kite-drp
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
IPAddressAllow=localhost
UMask=0077
# 硬件看门狗（树莓派启用后可选）
# WatchdogSec=30

[Install]
WantedBy=multi-user.target
```

启用与开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kite-drp-server.service
systemctl status kite-drp-server --no-pager
journalctl -u kite-drp-server -f          # 日志（无 PHI/输入数据）
curl -s http://127.0.0.1:8000/health       # 探活
```

> 说明：`IPAddressAllow=localhost` + `KITE_SERVER_HOST=127.0.0.1` 双重限制：服务只能被
> 本机访问，符合「装置全离线、数据不出院」（计划书 §9.4）。**不要**改成 `0.0.0.0`。

可选的健康巡检（每 5 分钟记录一次状态，失败重启）：

```ini
# /etc/systemd/system/kite-drp-healthcheck.service + .timer（示例）
[Service]
Type=oneshot
ExecStart=/usr/bin/curl -fsS http://127.0.0.1:8000/health -o /var/lib/kite-drp/last_health.json

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
```

---

## 3. 双目录原子替换（升级）

以 `0.1.0 → 0.2.0` 为例（**离线**：U 盘拷贝新 release + 签名文件）：

```bash
# 1) 校验新版本整包（SHA256SUMS + 分离签名，见 §5）
cd /media/usb/kite-drp-offline-0.2.0
sha256sum -c SHA256SUMS
minisign -Vm SHA256SUMS -p /etc/kite-drp/kite-drp.pub      # 失败则终止

# 2) 解包到 releases/0.2.0（只读、root 所有）
sudo mkdir -p /opt/kite-drp/releases/0.2.0
sudo tar -xzf kite-drp-offline-0.2.0.tar.gz -C /opt/kite-drp/releases/0.2.0
sudo chown -R root:kite /opt/kite-drp/releases/0.2.0
sudo chmod -R a-w /opt/kite-drp/releases/0.2.0

# 3) 校验部署包（manifest + model SHA256）
/opt/kite-drp/releases/0.2.0/server/../deploy/edge/scripts/verify_package.sh \
  /opt/kite-drp/releases/0.2.0/packages/current

# 4) 原子切换 current 软链（失败自动保留旧版）
sudo ln -sfn /opt/kite-drp/releases/0.2.0 /opt/kite-drp/current.new
sudo mv -Tf /opt/kite-drp/current.new /opt/kite-drp/current
sudo ln -sfn /opt/kite-drp/releases/0.1.0 /opt/kite-drp/previous.new
sudo mv -Tf /opt/kite-drp/previous.new /opt/kite-drp/previous

# 5) 重启并验收（≤2 分钟回滚窗口从这一步开始计时）
sudo systemctl restart kite-drp-server
curl -fsS http://127.0.0.1:8000/health
curl -fsS -X POST http://127.0.0.1:8000/api/v1/jobs -H 'Content-Type: application/json' \
  -d '{"sample_id":"SMOKE-0001","drug":"Osimertinib","mutation":{"EGFR":1}}' >/dev/null
```

**回滚（目标 ≤2 分钟）**：

```bash
sudo ln -sfn /opt/kite-drp/releases/0.1.0 /opt/kite-drp/current.new
sudo mv -Tf /opt/kite-drp/current.new /opt/kite-drp/current
sudo systemctl restart kite-drp-server
curl -fsS http://127.0.0.1:8000/health            # 确认 model_version 回到 0.1.0
```

保留策略：`releases/` 至少保留当前与上一版（`previous` 软链），旧版本清理放入二期脚本；
回滚演练建议在每次升级后由运维执行一次（计划书 §9.3 #11）。

---

## 4. 开机自启与断电恢复

- `systemctl enable` 已覆盖开机自启；服务 `Restart=on-failure` 处理崩溃自动拉起；
- 树莓派建议启用硬件看门狗（`/boot/firmware/config.txt` 的 `dtparam=watchdog=on` +
  `systemd` 的 `RuntimeWatchdogSec=`），断电恢复后自动回到上次版本（`current` 软链持久化）；
- 低功耗场景可设置 `RuntimeWatchdogSec=30` 与主动散热策略（计划书 §9.1/§9.3 #9）；
- 医生站 Windows 备选：用**任务计划程序**（触发器=登录/开机，操作=`uv.exe run uvicorn ...`
  或 `server\.venv\Scripts\python.exe -m uvicorn ...`，起始于 `deploy\server` 目录）
  或 NSSM 注册为 Windows 服务；同样只绑定 `127.0.0.1`。

---

## 5. 签名校验方案（计划书 §9.3 #11）

目标：**签名验签失败拒绝加载/拒绝安装**；模型 SHA256 已由服务启动时强制校验。

一期建议（简单、离线可验证）：

1. 训练机发布：`tar` 打包离线 bundle → 生成 `SHA256SUMS` → 用 **Ed25519 分离签名**
   （`minisign -Sm SHA256SUMS` 或 `gpg --detach-sign`）→ 同 U 盘拷贝；
2. 装置安装：先 `sha256sum -c SHA256SUMS`，再验签（公钥固化在
   `/etc/kite-drp/kite-drp.pub`，权限 0644；**私钥永不进装置**）；
3. 任一步失败 → 终止安装/升级，保留旧版本继续运行（`current` 不动）；
4. 服务启动时再做一次模型 SHA256 校验（防运行期被替换），失败回退 demo 模式并高亮告警。

二期可升级为：sigstore/cosign 透明日志、SBOM + 依赖哈希、装置 TPM 绑定密钥。

---

## 6. 日志、隐私与运维检查单

| 检查项 | 要求 |
|---|---|
| 日志内容 | 只含 job_id/状态/耗时/模型版本；**无 PHI、无原始基因值**（服务已实现） |
| 日志去向 | journald（`journalctl -u kite-drp-server`）；不落用户目录、不外发 |
| 数据留存 | 任务/结果仅在内存，重启清空；装置不存 FASTQ/BAM 等原始测序数据（§9.4） |
| 网络 | `ss -tnp` 无外部连接；`IPAddressAllow=localhost`；防火墙默认拒绝入站 |
| 磁盘 | 部署包只读；`/var/lib/kite-drp` 仅日志/巡检文件；建议全盘加密（§9.4） |
| 版本 | `current` 软链 + `previous` 保留；`/health.model_version` 可核对 |
| 依赖 | 装置不装 PyTorch；只装 ONNX Runtime CPU EP（aarch64 wheel，§9.2） |
| 回滚演练 | 每次升级后执行 §3 回滚流程，记录耗时（目标 ≤2 分钟） |

---

## 7. 一期 / 二期范围

- **一期（文档化，已完成）**：目录布局、systemd unit、双目录原子替换与回滚流程、
  签名校验方案、运维检查单；
- **二期（脚本化）**：`kite-drp-device` CLI（install/upgrade/rollback/verify）、自动保留
  N 个 release、A/B 分区、健康巡检 timer 自动安装、Windows 服务化脚本。

## 免责声明

设备侧部署与运维须遵守医院信息安全、伦理审查与《个人信息保护法》《人类遗传资源
管理条例》；本目录为科研原型规划，不用于临床决策（总计划书 §9.4）。
