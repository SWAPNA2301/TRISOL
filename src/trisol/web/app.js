// Trisol live dashboard. Vanilla JS, no build step, no framework.
//
// The server pushes the whole state over server-sent events whenever anything
// changes (a check starts, a check finishes, a fix lands), and every view is a
// pure function of that state. There is no client-side polling and no state
// the page can get out of step with.

"use strict";

let STATE = null;
let filter = "all";
const selectedFixes = new Set();

const SEVERITIES = ["critical", "high", "medium", "low"];
const RUBRIC_LABELS = {
  substantive: "Substantive",
  specific_role: "Specific role",
  output_contract: "Output contract",
  refusal_path: "Refusal path",
  grounding: "Grounding",
  concrete: "Concrete",
};

// ---- small helpers -------------------------------------------------------

function esc(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function $(id) {
  return document.getElementById(id);
}

function scoreClass(score) {
  if (score == null) return "s-na";
  if (score >= 80) return "s-good";
  if (score >= 50) return "s-mid";
  return "s-bad";
}

function ms(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`;
}

function svg(tag, attrs, children = "") {
  const a = Object.entries(attrs)
    .map(([k, v]) => `${k}="${esc(v)}"`)
    .join(" ");
  return `<${tag} ${a}>${children}</${tag}>`;
}

// ---- connection ----------------------------------------------------------

function connect() {
  // ?snapshot renders the current state once with no live stream -- for
  // printing, saving the page, or headless screenshots, none of which finish
  // while a server-sent-events connection is held open.
  if (new URLSearchParams(window.location.search).has("snapshot") || !window.EventSource) {
    fetch("/api/state", { cache: "no-store" })
      .then((res) => res.json())
      .then((state) => {
        STATE = state;
        render();
      });
    return;
  }
  const source = new EventSource("/events");
  source.onmessage = (event) => {
    STATE = JSON.parse(event.data);
    render();
  };
  source.onerror = () => {
    setStatus("error", "disconnected - retrying");
  };
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);
  return data;
}

// ---- top bar -------------------------------------------------------------

function setStatus(kind, text) {
  const el = $("status");
  el.className = `status status-${kind}`;
  $("status-text").textContent = text;
}

function renderHeader() {
  $("target").textContent = STATE.target;
  const s = STATE.status;
  if (s === "running") setStatus("running", `running audit #${STATE.run}`);
  else if (s === "done") setStatus("done", `audit #${STATE.run} complete`);
  else if (s === "error") setStatus("error", "audit failed");
  else setStatus("idle", "waiting");
  $("rerun").disabled = s === "running";
  const report = STATE.report;
  $("count-findings").textContent = report ? report.summary.total : "";
  $("count-fixes").textContent = STATE.fixes.length || "";
  $("footer-version").textContent = report ? `trisol ${report.trisol_version}` : "";
}

// ---- overview ------------------------------------------------------------

function renderScore() {
  const report = STATE.report;
  const card = report ? report.scorecard : null;
  const overall = card ? card.overall : null;
  const fill = $("ring-fill");
  const circumference = 2 * Math.PI * 50;
  fill.style.strokeDasharray = `${circumference}`;
  fill.style.strokeDashoffset = `${circumference * (1 - (overall || 0) / 100)}`;
  fill.setAttribute("class", `ring-fill ${scoreClass(overall)}`);
  $("score").textContent = overall == null ? "--" : overall;

  const summary = report ? report.summary : { critical: 0, high: 0, medium: 0, low: 0 };
  $("sev-row").innerHTML = SEVERITIES.map(
    (sev) => `<div class="sev sev-${sev}"><b>${summary[sev] || 0}</b><span>${sev}</span></div>`,
  ).join("");

  $("score-hint").textContent =
    STATE.status === "running"
      ? "Audit in progress - the score updates when it finishes."
      : overall == null
        ? "No agent code found to score."
        : "Mean of the measured areas below. Every number comes from a check.";
}

function renderPipeline() {
  const checks = STATE.checks || [];
  const icon = { pending: "", running: "", done: "", skipped: "" };
  $("pipeline").innerHTML = checks
    .map((c) => {
      let detail = "waiting";
      if (c.state === "running") detail = "running...";
      if (c.state === "done") detail = `${c.findings} finding${c.findings === 1 ? "" : "s"} &middot; ${ms(c.duration_ms)}`;
      if (c.state === "skipped") detail = "not applicable";
      return `<li class="step step-${c.state}" title="${esc(c.skip_reason || "")}">
        <span class="step-icon">${icon[c.state] || ""}</span>
        <span class="step-name">${esc(c.title)}</span>
        <span class="step-detail">${detail}</span>
      </li>`;
    })
    .join("");
  const finished = checks.filter((c) => c.state === "done" || c.state === "skipped").length;
  const pct = checks.length ? (100 * finished) / checks.length : 0;
  $("progress").style.width = `${STATE.status === "done" ? 100 : pct}%`;
  $("run-label").textContent = STATE.run ? `run #${STATE.run}` : "";
}

function renderAreas() {
  const report = STATE.report;
  const areas = report ? report.scorecard.areas : [];
  if (!areas.length) {
    $("areas").innerHTML = `<p class="empty">${
      STATE.status === "running" ? "Waiting for the audit to finish..." : "Nothing measurable in this target."
    }</p>`;
    return;
  }
  $("areas").innerHTML = areas
    .map(
      (a) => `<div class="area">
        <div class="area-head"><span>${esc(a.title)}</span><b class="${scoreClass(a.score)}">${a.score}</b></div>
        <div class="bar"><div class="bar-fill ${scoreClass(a.score)}" style="width:${a.score}%"></div></div>
        <div class="area-note">${esc(a.summary)}</div>
      </div>`,
    )
    .join("");
}

function renderHistory() {
  const points = (STATE.history || []).filter((h) => h.overall != null);
  if (points.length < 1) {
    $("history").innerHTML = `<p class="empty">The first run will appear here.</p>`;
    return;
  }
  const w = 520;
  const h = 170;
  const pad = { l: 34, r: 14, t: 14, b: 26 };
  const x = (i) =>
    points.length === 1 ? (pad.l + w - pad.r) / 2 : pad.l + (i * (w - pad.l - pad.r)) / (points.length - 1);
  const y = (v) => pad.t + (1 - v / 100) * (h - pad.t - pad.b);

  let grid = "";
  for (const v of [0, 50, 100]) {
    grid += svg("line", { x1: pad.l, x2: w - pad.r, y1: y(v), y2: y(v), class: "grid" });
    grid += svg("text", { x: pad.l - 6, y: y(v) + 4, class: "axis", "text-anchor": "end" }, v);
  }
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.overall)}`).join(" ");
  const area = `${path} L${x(points.length - 1)},${y(0)} L${x(0)},${y(0)} Z`;
  let dots = "";
  points.forEach((p, i) => {
    dots += svg("circle", { cx: x(i), cy: y(p.overall), r: 4.5, class: `dot ${scoreClass(p.overall)}` });
    dots += svg("text", { x: x(i), y: y(p.overall) - 10, class: "dot-label", "text-anchor": "middle" }, p.overall);
    dots += svg("text", { x: x(i), y: h - 6, class: "axis", "text-anchor": "middle" }, `#${p.run}`);
  });
  $("history").innerHTML = `<svg viewBox="0 0 ${w} ${h}" class="history-svg">${grid}
    <path d="${area}" class="area-fill"/><path d="${path}" class="line"/>${dots}</svg>`;
}

// ---- findings ------------------------------------------------------------

function allFindings() {
  if (!STATE.report) return [];
  const out = [];
  for (const r of STATE.report.results) for (const f of r.findings) out.push(f);
  const conf = { certain: 0, likely: 1, possible: 2 };
  out.sort(
    (a, b) =>
      SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity) ||
      conf[a.confidence] - conf[b.confidence] ||
      String(a.file || "").localeCompare(String(b.file || "")) ||
      (a.line || 0) - (b.line || 0),
  );
  return out;
}

function renderFindings() {
  const findings = allFindings().filter((f) => {
    if (filter === "all") return true;
    if (filter === "fixable") return f.fixable;
    if (filter === "claude") return f.source === "claude";
    return f.severity === filter;
  });
  if (!STATE.report) {
    $("findings").innerHTML = `<p class="empty card">Findings appear as soon as the audit finishes.</p>`;
    return;
  }
  if (!findings.length) {
    $("findings").innerHTML = `<p class="empty card">Nothing matches this filter.</p>`;
    return;
  }
  $("findings").innerHTML = findings
    .map((f) => {
      const where = f.file ? `<span class="where mono">${esc(f.file)}${f.line ? ":" + f.line : ""}</span>` : "";
      const tags = [];
      if (f.source === "claude") tags.push(`<span class="tag tag-claude">claude</span>`);
      if (f.confidence !== "certain") tags.push(`<span class="tag">${esc(f.confidence)}</span>`);
      if (f.fixable) tags.push(`<span class="tag tag-fix">auto-fixable</span>`);
      return `<article class="finding sev-${f.severity}">
        <header><span class="badge">${esc(f.severity)}</span><h3>${esc(f.title)}</h3>${where}<span class="grow"></span>${tags.join("")}</header>
        <p>${esc(f.detail)}</p>
        ${f.impact ? `<p class="kv"><b>Impact</b>${esc(f.impact)}</p>` : ""}
        ${f.suggestion ? `<p class="kv"><b>Fix</b>${esc(f.suggestion)}</p>` : ""}
      </article>`;
    })
    .join("");
}

// ---- benchmarks ----------------------------------------------------------

function benchRetrieval(area) {
  // HTML bars rather than SVG: an SVG chart scales its text with the card, so
  // labels grew to twice their size on wide screens.
  const rows = area.strategies
    .map((r) => {
      const isBest = r.name === area.best;
      const isYours = r.name === area.baseline;
      const cls = isBest ? "s-good" : isYours ? "s-bad" : "s-na";
      const tag = isBest
        ? `<span class="tag tag-best">best</span>`
        : isYours
          ? `<span class="tag tag-yours">your code</span>`
          : "";
      return `<div class="strategy">
        <div class="strategy-name">${esc(r.name)} ${tag}</div>
        <div class="strategy-bars">
          <div class="bar"><div class="bar-fill ${cls}" style="width:${Math.max(1, r.recall_at_k * 100)}%"></div></div>
          <div class="bar thin"><div class="bar-fill mrr" style="width:${Math.max(1, r.mrr * 100)}%"></div></div>
        </div>
        <div class="strategy-nums"><b class="${cls}">${Math.round(r.recall_at_k * 100)}%</b><span class="muted">MRR ${r.mrr.toFixed(2)}</span></div>
      </div>`;
    })
    .join("");
  return `<div class="card">
    <div class="card-title">Retrieval strategies <span class="muted">recall@k and MRR on this project's own corpus</span></div>
    <div class="legend"><span><i class="sw s-good"></i>recall@k</span><span><i class="sw mrr"></i>mean reciprocal rank</span></div>
    ${rows}
    <p class="hint">${esc(area.summary)}</p>
  </div>`;
}

function benchPrompts(area) {
  const keys = Object.keys(RUBRIC_LABELS);
  const head = keys.map((k) => `<th>${RUBRIC_LABELS[k]}</th>`).join("");
  const rows = area.prompts
    .map((p) => {
      const cells = keys
        .map((k) => {
          const v = p.criteria[k];
          if (v === null || v === undefined) return `<td class="cell na" title="not applicable">n/a</td>`;
          return `<td class="cell ${v ? "pass" : "fail"}">${v ? "pass" : "fail"}</td>`;
        })
        .join("");
      return `<tr><td class="mono pname">${esc(p.name)}<div class="muted">${esc(p.file)}:${p.line}</div></td>${cells}
        <td><div class="mini-bar"><div class="bar-fill ${scoreClass(p.score)}" style="width:${p.score}%"></div></div><b class="${scoreClass(p.score)}">${p.score}</b></td></tr>`;
    })
    .join("");
  return `<div class="card">
    <div class="card-title">Prompt rubric <span class="muted">each criterion is a checkable property of the text</span></div>
    <div class="table-wrap"><table class="rubric"><thead><tr><th>Prompt</th>${head}<th>Score</th></tr></thead><tbody>${rows}</tbody></table></div>
  </div>`;
}

function benchBars(title, note, bars) {
  const rows = bars
    .map((b) => {
      const pct = b.max ? (100 * b.value) / b.max : 0;
      return `<div class="area">
        <div class="area-head"><span>${esc(b.label)}</span><b class="${scoreClass(pct)}">${b.value} / ${b.max}</b></div>
        <div class="bar"><div class="bar-fill ${scoreClass(pct)}" style="width:${pct}%"></div></div>
      </div>`;
    })
    .join("");
  return `<div class="card"><div class="card-title">${esc(title)} <span class="muted">${esc(note)}</span></div>${rows}</div>`;
}

function benchRouting(area) {
  const agents = area.agents || [];
  const max = Math.max(1, ...agents.map((a) => a.keywords));
  const rows = agents
    .map(
      (a) => `<div class="stack-row">
        <span class="label">${esc(a.agent)}</span>
        <div class="stack">
          <div class="stack-part s-good" style="width:${(100 * a.unique) / max}%" title="${a.unique} unique"></div>
          <div class="stack-part s-bad" style="width:${(100 * a.shared) / max}%" title="${a.shared} shared"></div>
        </div>
        <span class="muted">${a.unique} unique &middot; ${a.shared} shared</span>
      </div>`,
    )
    .join("");
  return `<div class="card">
    <div class="card-title">Routing keywords <span class="muted">shared keywords make the route depend on dict order</span></div>
    ${rows}
    <p class="hint">${esc(area.summary)}</p>
  </div>`;
}

function renderBench() {
  const report = STATE.report;
  if (!report) {
    $("bench").innerHTML = `<p class="empty card">Benchmarks appear when the audit finishes.</p>`;
    return;
  }
  const areas = Object.fromEntries(report.scorecard.areas.map((a) => [a.key, a]));
  const parts = [];
  if (areas.retrieval) parts.push(benchRetrieval(areas.retrieval));
  if (areas.prompts) parts.push(benchPrompts(areas.prompts));
  if (areas.safeguards) parts.push(benchBars("Model-call safeguards", areas.safeguards.summary, areas.safeguards.bars));
  if (areas.routing) parts.push(benchRouting(areas.routing));
  if (areas.data) parts.push(benchBars("Knowledge base", areas.data.summary, areas.data.bars || []));
  $("bench").innerHTML = parts.length
    ? parts.join("")
    : `<p class="empty card">No prompts, model calls, corpus or routing table found to measure.</p>`;
}

// ---- fixes ---------------------------------------------------------------

function renderFixes() {
  const fixes = STATE.fixes || [];
  for (const id of [...selectedFixes]) if (!fixes.some((f) => f.id === id)) selectedFixes.delete(id);
  const busy = STATE.status === "running";
  $("fix-all").disabled = busy || !fixes.length;
  $("fix-selected").disabled = busy || !selectedFixes.size;

  const last = STATE.last_fix;
  $("fix-result").innerHTML = last
    ? `<div class="card fix-done">
        <b>${last.applied.length} fix${last.applied.length === 1 ? "" : "es"} applied</b>
        ${last.refused.length ? `&middot; ${last.refused.length} refused` : ""}
        <span class="muted"> &middot; the audit re-ran to verify; see the score history</span>
        ${last.refused.map((r) => `<div class="refused">${esc(r.title)}: ${esc(r.reason)}</div>`).join("")}
      </div>`
    : "";

  if (!fixes.length) {
    $("fixes").innerHTML = `<p class="empty card">${
      STATE.report ? "No automatic fixes for the current findings." : "Waiting for the audit..."
    }</p>`;
    return;
  }
  $("fixes").innerHTML = fixes
    .map(
      (f) => `<article class="fix sev-${f.severity}">
        <label class="fix-head">
          <input type="checkbox" data-id="${f.id}" ${selectedFixes.has(f.id) ? "checked" : ""}>
          <span class="badge">${esc(f.severity)}</span>
          <b>${esc(f.title)}</b>
          <span class="where mono">${esc(f.file)}${f.line ? ":" + f.line : ""}</span>
        </label>
        <pre class="diff"><span class="del">- ${esc(f.old)}</span><span class="add">+ ${esc(f.new)}</span></pre>
        <p class="hint">${esc(f.explanation)}</p>
        <button class="btn small" data-apply="${f.id}" ${busy ? "disabled" : ""}>Apply this fix</button>
      </article>`,
    )
    .join("");

  document.querySelectorAll("#fixes input[type=checkbox]").forEach((box) =>
    box.addEventListener("change", () => {
      const id = Number(box.dataset.id);
      if (box.checked) selectedFixes.add(id);
      else selectedFixes.delete(id);
      $("fix-selected").disabled = busy || !selectedFixes.size;
    }),
  );
  document.querySelectorAll("#fixes [data-apply]").forEach((btn) =>
    btn.addEventListener("click", () => applyFixes([Number(btn.dataset.apply)])),
  );
}

async function applyFixes(ids) {
  const count = ids ? ids.length : (STATE.fixes || []).length;
  if (!window.confirm(`Write ${count} fix${count === 1 ? "" : "es"} into your files?`)) return;
  try {
    await post("/api/fix", ids ? { ids } : {});
    selectedFixes.clear();
  } catch (err) {
    window.alert(err.message);
  }
}

// ---- activity ------------------------------------------------------------

function renderLog() {
  const log = STATE.log || [];
  const el = $("log");
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  el.innerHTML = log
    .map((entry) => {
      const t = new Date(entry.t * 1000).toLocaleTimeString();
      return `<div class="log-${esc(entry.kind)}"><span class="muted">${t}</span>  ${esc(entry.text)}</div>`;
    })
    .join("");
  if (atBottom) el.scrollTop = el.scrollHeight;
}

// ---- render --------------------------------------------------------------

function render() {
  if (!STATE) return;
  renderHeader();
  renderScore();
  renderPipeline();
  renderAreas();
  renderHistory();
  renderFindings();
  renderBench();
  renderFixes();
  renderLog();
}

// ---- wiring --------------------------------------------------------------

function showView(name) {
  const tab = document.querySelector(`#tabs .tab[data-view="${name}"]`);
  if (!tab) return;
  document.querySelectorAll("#tabs .tab").forEach((t) => t.classList.remove("is-active"));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
  tab.classList.add("is-active");
  $(`view-${name}`).classList.add("is-active");
}

document.querySelectorAll("#tabs .tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    showView(tab.dataset.view);
    // Linkable tabs: the URL hash names the open view.
    history.replaceState(null, "", `#${tab.dataset.view}`);
  }),
);
if (window.location.hash) showView(window.location.hash.slice(1));

document.querySelectorAll("#chips .chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    document.querySelectorAll("#chips .chip").forEach((c) => c.classList.remove("is-active"));
    chip.classList.add("is-active");
    filter = chip.dataset.filter;
    renderFindings();
  }),
);

$("rerun").addEventListener("click", () => post("/api/run").catch((err) => window.alert(err.message)));
$("fix-all").addEventListener("click", () => applyFixes(null));
$("fix-selected").addEventListener("click", () => applyFixes([...selectedFixes]));

(function theme() {
  let saved = null;
  try {
    saved = localStorage.getItem("trisol-theme");
  } catch (err) {
    /* storage blocked: follow the OS */
  }
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const now = document.documentElement.dataset.theme || (dark ? "dark" : "light");
    const next = now === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("trisol-theme", next);
    } catch (err) {
      /* not persisted */
    }
  });
})();

connect();
