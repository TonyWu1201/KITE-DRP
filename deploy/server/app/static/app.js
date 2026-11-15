/* =============================================================================
 * KITE-DRP 本机推理前端
 *
 * 合规约束（计划书 §9.3 #10、§9.4）：
 *   - 只允许同源相对请求（fetch("/health")、fetch("/api/v1/...")），
 *     无 CDN / 外部字体 / 上报 / 任何运行期外网请求；
 *   - 所有用户输入只用 textContent 写入 DOM（不使用 innerHTML），避免 XSS；
 *   - 页面顶部固定展示「科研原型，不用于临床决策」，demo 模式时额外展示红色横幅。
 * ========================================================================== */
(function () {
  "use strict";

  var API = {
    health: "/health",
    jobs: "/api/v1/jobs",
  };
  var TERMINAL = ["succeeded", "failed"];
  var STATUS_TEXT = {
    queued: "排队中",
    running: "推理中",
    succeeded: "已完成",
    failed: "失败",
  };
  var POLL_INTERVAL_MS = 800;
  var HEALTH_INTERVAL_MS = 15000;
  var MAX_POLL_MS = 60000;

  var EXAMPLE = {
    schema_version: "1.0",
    sample_id: "DEMO-0001",
    assay: "DNA panel + RNA-seq",
    mutation: { EGFR: 1, TP53: 1, KRAS: 0 },
    expression: { EGFR: 5.2, TP53: 4.1, MYC: 6.0 },
    drug: "Osimertinib",
    meta: { cancer_type: "LUAD", stage: "IV" },
  };

  var pollTimer = null;
  var pollStartedAt = 0;
  var currentJobId = null;

  function el(id) {
    return document.getElementById(id);
  }

  function show(node, visible) {
    node.hidden = !visible;
  }

  function createElement(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function shortId(jobId) {
    return typeof jobId === "string" && jobId.length > 12 ? jobId.slice(0, 12) + "…" : String(jobId);
  }

  function formatTime(iso) {
    if (!iso) {
      return "—";
    }
    var date = new Date(iso);
    if (Number.isNaN(date.getTime())) {
      return String(iso);
    }
    return date.toLocaleString("zh-CN", { hour12: false });
  }

  function extractErrorMessage(body, fallback) {
    if (!body || !body.detail) {
      return fallback;
    }
    var detail = body.detail;
    if (Array.isArray(detail)) {
      return detail
        .map(function (item) {
          var location = Array.isArray(item.loc) ? item.loc.join(".") : "";
          return (location ? location + "：" : "") + (item.msg || "校验失败");
        })
        .join("；");
    }
    if (typeof detail === "string") {
      return detail;
    }
    if (detail.message) {
      return detail.message;
    }
    return fallback;
  }

  function fetchJSON(url, options) {
    var init = {
      method: (options && options.method) || "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    };
    if (options && options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    return fetch(url, init).then(function (response) {
      return response
        .json()
        .catch(function () {
          return null;
        })
        .then(function (body) {
          if (!response.ok) {
            throw new Error(extractErrorMessage(body, "HTTP " + response.status));
          }
          return body;
        });
    });
  }

  // ---------------------------------------------------------------- 输入解析
  function splitTokens(text) {
    return text.split(/[\s,;，；]+/).filter(function (token) {
      return token.length > 0;
    });
  }

  function parseJSONObject(text, label) {
    var trimmed = text.trim();
    if (!trimmed) {
      return null;
    }
    var value;
    try {
      value = JSON.parse(trimmed);
    } catch (err) {
      throw new Error(label + "不是合法 JSON：" + err.message);
    }
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(label + "必须是 JSON 对象（{...}）");
    }
    return value;
  }

  function parseMutationText(text) {
    var trimmed = text.trim();
    if (!trimmed) {
      return {};
    }
    if (trimmed.charAt(0) === "{") {
      return parseJSONObject(trimmed, "突变");
    }
    var result = {};
    splitTokens(trimmed).forEach(function (token) {
      var index = token.indexOf(":");
      if (index < 0) {
        result[token] = 1;
        return;
      }
      var gene = token.slice(0, index).trim();
      var flag = token.slice(index + 1).trim();
      if (!gene) {
        throw new Error("突变条目缺少基因名：" + token);
      }
      if (flag !== "0" && flag !== "1") {
        throw new Error("突变 " + gene + " 的值必须是 0 或 1（收到 " + flag + "）");
      }
      result[gene] = Number(flag);
    });
    return result;
  }

  function parseExpressionText(text) {
    var trimmed = text.trim();
    if (!trimmed) {
      return {};
    }
    if (trimmed.charAt(0) === "{") {
      return parseJSONObject(trimmed, "表达");
    }
    var result = {};
    splitTokens(trimmed).forEach(function (token) {
      var index = token.indexOf(":");
      if (index < 0) {
        throw new Error("表达条目需写成 基因:数值（例如 EGFR:5.2），收到：" + token);
      }
      var gene = token.slice(0, index).trim();
      var rawValue = token.slice(index + 1).trim();
      var value = Number(rawValue);
      if (!gene || rawValue === "" || Number.isNaN(value) || !Number.isFinite(value)) {
        throw new Error("表达 " + token + " 格式非法（应为 基因:有限数值）");
      }
      result[gene] = value;
    });
    return result;
  }

  function buildPayload() {
    var sampleId = el("sample-id").value.trim();
    var drug = el("drug").value.trim();
    if (!sampleId) {
      throw new Error("请填写样本 ID（伪名化）");
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(sampleId)) {
      throw new Error("样本 ID 只允许字母/数字/._-，且不以符号开头（≤64 字符）");
    }
    if (!drug) {
      throw new Error("请填写药物名称");
    }
    var mutation = parseMutationText(el("mutation").value);
    var expression = parseExpressionText(el("expression").value);
    if (Object.keys(mutation).length === 0 && Object.keys(expression).length === 0) {
      throw new Error("突变与表达至少填写一项");
    }
    var payload = {
      schema_version: "1.0",
      sample_id: sampleId,
      mutation: mutation,
      expression: expression,
      drug: drug,
      meta: parseJSONObject(el("meta").value, "临床元数据") || {},
    };
    var assay = el("assay").value.trim();
    if (assay) {
      payload.assay = assay;
    }
    return payload;
  }

  // ---------------------------------------------------------------- 健康状态
  function refreshHealth() {
    return fetchJSON(API.health)
      .then(function (health) {
        var pill = el("health-pill");
        if (health.status === "ok") {
          pill.textContent = "ONNX 模型就绪";
          pill.className = "pill pill-ok";
        } else {
          pill.textContent = "DEMO 模式";
          pill.className = "pill pill-warn";
        }
        el("health-meta").textContent =
          "模型 " +
          health.model_version +
          " · 队列 " +
          health.queue.size +
          "/" +
          health.queue.capacity +
          " · " +
          (health.demo ? "demo" : "onnx");
        var banner = el("demo-banner");
        if (health.demo) {
          banner.hidden = false;
          el("demo-reason").textContent = health.demo_reason ? "（原因：" + health.demo_reason + "）" : "";
        } else {
          banner.hidden = true;
        }
      })
      .catch(function (err) {
        var pill = el("health-pill");
        pill.textContent = "服务未连接";
        pill.className = "pill pill-err";
        el("health-meta").textContent = err.message;
      });
  }

  // ---------------------------------------------------------------- 任务展示
  function showFormError(message) {
    var node = el("form-error");
    node.textContent = message;
    show(node, true);
  }

  function hideFormError() {
    show(el("form-error"), false);
  }

  function setBusy(busy) {
    var button = el("submit-btn");
    button.disabled = busy;
    button.textContent = busy ? "推理中…" : "提交预测任务";
    el("parse-hint").textContent = busy && currentJobId ? "任务 " + shortId(currentJobId) : "";
  }

  function renderJob(job) {
    currentJobId = job.job_id;
    var stateText = "任务 " + shortId(job.job_id) + " · " + (STATUS_TEXT[job.status] || job.status);
    if (job.status === "queued") {
      stateText += " · 已入队，等待 worker";
    } else if (job.status === "running") {
      stateText += " · 正在推理";
    } else if (job.status === "succeeded") {
      stateText += " · 完成于 " + formatTime(job.finished_at) + " · 耗时 " + job.result.elapsed_ms + " ms";
    } else {
      stateText += " · 失败于 " + formatTime(job.finished_at) + " · " + (job.error || "未知错误");
    }
    el("job-state").textContent = stateText;

    if (job.status === "succeeded" && job.result) {
      renderResult(job.result);
    } else if (job.status === "failed") {
      renderResult(null, job.error);
    } else {
      show(el("result"), false);
    }
  }

  function renderResult(result, errorMessage) {
    var box = el("result");
    show(box, true);
    var warnings = el("res-warnings");
    warnings.textContent = "";
    var explanation = el("res-explanation");
    explanation.textContent = "";

    if (!result) {
      el("res-score").textContent = "—";
      el("res-ci").textContent = "任务失败，未产生结果";
      el("res-sample").textContent = "";
      el("res-model").textContent = "—";
      el("res-sha").textContent = "—";
      el("res-engine").textContent = "—";
      el("res-elapsed").textContent = "—";
      show(el("res-demo"), false);
      show(el("res-explain-wrap"), false);
      warnings.appendChild(createElement("li", null, errorMessage || "推理失败（详情见服务端日志）"));
      return;
    }

    el("res-score").textContent = result.response.toFixed(4);
    el("res-ci").textContent =
      String(Math.round(result.ci.level * 100)) +
      "% CI：" +
      result.ci.low.toFixed(4) +
      " – " +
      result.ci.high.toFixed(4) +
      "（科研用未校准区间）";
    el("res-sample").textContent = "样本 " + result.sample_id + " · 药物 " + result.drug;
    el("res-model").textContent = result.model_version + (result.demo ? "（DEMO）" : "");
    el("res-sha").textContent = result.model_sha256;
    el("res-engine").textContent = result.engine;
    el("res-elapsed").textContent = result.elapsed_ms.toFixed(2) + " ms";
    show(el("res-demo"), Boolean(result.demo));
    result.warnings.forEach(function (warning) {
      warnings.appendChild(createElement("li", null, warning));
    });

    var topFeatures = result.explanation && result.explanation.top_features;
    if (Array.isArray(topFeatures) && topFeatures.length > 0) {
      topFeatures.forEach(function (feature) {
        explanation.appendChild(
          createElement("li", null, feature.gene + " · contribution " + feature.contribution)
        );
      });
      show(el("res-explain-wrap"), true);
    } else {
      show(el("res-explain-wrap"), false);
    }
  }

  function stopPolling() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function fetchJob(jobId) {
    return fetchJSON(API.jobs + "/" + encodeURIComponent(jobId));
  }

  function pollJob(jobId) {
    stopPolling();
    fetchJob(jobId)
      .then(function (job) {
        renderJob(job);
        if (TERMINAL.indexOf(job.status) >= 0) {
          setBusy(false);
          stopPolling();
          refreshRecent();
          refreshHealth();
          return;
        }
        if (Date.now() - pollStartedAt > MAX_POLL_MS) {
          setBusy(false);
          showFormError("轮询超时：任务仍在排队/推理中，请稍后在最近任务中查看");
          return;
        }
        pollTimer = setTimeout(function () {
          pollJob(jobId);
        }, POLL_INTERVAL_MS);
      })
      .catch(function (err) {
        setBusy(false);
        showFormError("轮询任务失败：" + err.message);
      });
  }

  function startJob(jobId) {
    currentJobId = jobId;
    pollStartedAt = Date.now();
    hideFormError();
    setBusy(true);
    show(el("result"), false);
    el("job-state").textContent = "任务 " + shortId(jobId) + " · 已提交，排队中…";
    pollJob(jobId);
  }

  // ---------------------------------------------------------------- 表单提交
  function onSubmit(event) {
    event.preventDefault();
    hideFormError();
    var payload;
    try {
      payload = buildPayload();
    } catch (err) {
      showFormError(err.message);
      return;
    }
    setBusy(true);
    el("job-state").textContent = "正在提交任务…";
    fetchJSON(API.jobs, { method: "POST", body: payload })
      .then(function (ack) {
        startJob(ack.job_id);
        refreshRecent();
      })
      .catch(function (err) {
        setBusy(false);
        el("job-state").textContent = "尚未提交任务。";
        showFormError("提交失败：" + err.message);
      });
  }

  function fillExample() {
    var payload = EXAMPLE;
    el("sample-id").value = payload.sample_id;
    el("drug").value = payload.drug;
    el("assay").value = payload.assay;
    el("mutation").value = JSON.stringify(payload.mutation);
    el("expression").value = JSON.stringify(payload.expression);
    el("meta").value = JSON.stringify(payload.meta);
    hideFormError();
    el("parse-hint").textContent = "已填入示例（demo 模式可直接提交）";
  }

  // ---------------------------------------------------------------- 最近任务
  function statusBadge(status) {
    return createElement("span", "badge badge-" + status, STATUS_TEXT[status] || status);
  }

  function refreshRecent() {
    return fetchJSON(API.jobs + "?limit=10")
      .then(function (data) {
        var body = el("recent-body");
        body.textContent = "";
        if (!data.items || data.items.length === 0) {
          var emptyRow = document.createElement("tr");
          var emptyCell = createElement("td", "empty", "暂无任务");
          emptyCell.colSpan = 5;
          emptyRow.appendChild(emptyCell);
          body.appendChild(emptyRow);
          return;
        }
        data.items.forEach(function (job) {
          var row = document.createElement("tr");
          row.appendChild(createElement("td", "mono", shortId(job.job_id)));
          row.appendChild(createElement("td", null, job.input.sample_id));
          row.appendChild(createElement("td", null, job.input.drug));
          var statusCell = document.createElement("td");
          statusCell.appendChild(statusBadge(job.status));
          row.appendChild(statusCell);
          row.appendChild(createElement("td", null, formatTime(job.updated_at)));
          row.addEventListener("click", function () {
            loadJob(job.job_id);
          });
          body.appendChild(row);
        });
      })
      .catch(function () {
        /* 最近列表失败不阻塞主流程（服务未就绪时静默） */
      });
  }

  function loadJob(jobId) {
    stopPolling();
    hideFormError();
    fetchJob(jobId)
      .then(function (job) {
        renderJob(job);
        if (TERMINAL.indexOf(job.status) < 0) {
          startJob(job.job_id);
        } else {
          setBusy(false);
        }
        refreshRecent();
      })
      .catch(function (err) {
        showFormError("载入任务失败：" + err.message);
      });
  }

  // ---------------------------------------------------------------- 初始化
  function init() {
    el("job-form").addEventListener("submit", onSubmit);
    el("example-btn").addEventListener("click", fillExample);
    refreshHealth();
    refreshRecent();
    setInterval(refreshHealth, HEALTH_INTERVAL_MS);
    setInterval(refreshRecent, HEALTH_INTERVAL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
