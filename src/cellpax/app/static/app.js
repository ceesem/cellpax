/* The client. No dependencies, on purpose — a local scientific tool should not
   stop working because a CDN is unreachable or a version moved under it, and
   34k points render fine on a plain canvas.

   The organising idea is the same one that shapes the server: operations that
   are free once the linkage is cached (threshold, size floor, flow, scatter)
   update inline as you move the control, and operations that build something
   (sweep, restrict, embed, merge support) are submitted, watched, and marked
   with a "job" badge. */

const S = {
  state: null,
  level: null,        // selected level name
  scan: null,
  preview: null,
  support: null,
  cluster: null,      // selected cluster id
  colors: {},
  names: {},
  graphTypes: new Set(["knn", "snn_jaccard", "umap_fuzzy"]),
  neighbors: new Set([15, 30, 60]),
};

/* Palette comes from the server so swatches, scatter and neuroglancer links
   cannot drift apart. Defaults here only cover the moment before /api/state
   lands. See ngl.py for why the order is what it is. */
let PALETTE = ["#1f77b4","#ff7f0e","#17becf","#d62728","#bcbd22","#9467bd",
               "#2ca02c","#e377c2","#8c564b"];
let UNASSIGNED = "#4a4f5e";
const colorFor = id =>
  (id < 0 ? UNASSIGNED : (S.colors[id] || PALETTE[id % PALETTE.length]));

/* Staleness.

   Every derived panel is computed for one (level, threshold, size floor). When
   any of those changes, whatever is on screen describes something you are no
   longer looking at. Silently leaving it there is the worst failure this tool
   can have — it is a picture of another cohort with nothing to say so. Two
   mechanisms, because one is not enough:

   * every panel records the token it was rendered for, and anything whose token
     no longer matches is visibly greyed and marked "stale";
   * every async loader re-checks the token after its fetch returns and drops
     the result if the view moved on, so a slow response cannot paint itself
     into the wrong level.

   Switching levels additionally hard-clears them, because "stale" is the right
   label for an old cut of the same cohort and simply wrong for a different one. */

const DERIVED = ["#coverage", "#axis", "#scanChart", "#plateaus", "#support",
                 "#flowChart", "#mapCanvas", "#profileChart", "#heatmapChart",
                 "#fsCanvas", "#boundary", "#gradient"];

const viewToken = () =>
  `${S.level}|${S.preview?.distance_threshold ?? ""}|${S.preview?.min_cluster_size ?? ""}`;

function stamp(sel) {
  const node = document.querySelector(sel);
  if (node) { node.dataset.token = viewToken(); node.classList.remove("stale"); }
}

function markStale() {
  const now = viewToken();
  for (const node of document.querySelectorAll("[data-token]")) {
    node.classList.toggle("stale", node.dataset.token !== now);
  }
}

/** Wipe every derived view. Used when the cohort itself changes. */
function resetViews() {
  for (const sel of DERIVED) {
    const node = document.querySelector(sel);
    if (!node) continue;
    node.classList.remove("stale");
    delete node.dataset.token;
    if (node.tagName === "CANVAS") {
      const ctx = node.getContext("2d");
      ctx.clearRect(0, 0, node.width, node.height);
    } else node.innerHTML = "";
  }
  mapData = null;
  columnsCache = null;
  gradPick = new Set();
  const builder = document.querySelector("#childBuilder");
  if (builder) { builder.innerHTML = ""; delete builder.dataset.signature; }
  const commit = document.querySelector("#commitClusters");
  if (commit) commit.innerHTML = "";
  const detail = document.querySelector("#inspectorDetail");
  if (detail) detail.innerHTML = "";
}

const $  = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
};
const fmt = (v, d = 3) =>
  v === null || v === undefined ? "—"
  : typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(d))
  : v;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const err = new Error(data?.detail || res.statusText);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

/* ── job watching ───────────────────────────────────────────────────────── */

let pollTimer = null;
function watchJobs() {
  clearTimeout(pollTimer);
  const tick = async () => {
    try {
      const { jobs, busy, build } = await api("/api/jobs");
      if (build) {
        if (S.build === undefined) S.build = build;
        else if (build !== S.build) showStaleClient();
      }
      const pill = $("#jobPill");
      const active = jobs.find(j => j.status === "running" || j.status === "queued");
      const failed = jobs.find(j => j.status === "failed");
      if (active) {
        pill.className = "pill busy";
        pill.textContent = `${active.label}${active.message ? " — " + active.message : ""}` +
          (active.elapsed ? ` (${active.elapsed.toFixed(0)}s)` : "");
      } else if (failed && failed.id === jobs[0]?.id) {
        pill.className = "pill failed";
        pill.textContent = `failed: ${failed.error}`;
      } else {
        pill.className = "pill idle";
        pill.textContent = "idle";
      }
      const settled = !busy && S.wasBusy;
      S.wasBusy = busy;               // set before the refresh, never skipped
      if (settled) {
        try {
          await refreshAll();
        } catch (err) {
          // A render failure here used to be swallowed whole, which looked
          // exactly like "the job never finished": the pill went idle and
          // nothing else ever updated.
          pill.className = "pill failed";
          pill.textContent = `job finished but the refresh failed: ${err.message}`;
        }
      }
    } catch { /* server restarting; keep polling */ }
    pollTimer = setTimeout(tick, busyPoll());
  };
  tick();
}
const busyPoll = () => (S.wasBusy ? 700 : 2500);

/* ── top-level refresh ──────────────────────────────────────────────────── */

async function refreshAll() {
  S.state = await api("/api/state");
  if (S.state.palette) {
    PALETTE = S.state.palette.colors;
    UNASSIGNED = S.state.palette.unassigned;
  }
  $("#tableInfo").textContent =
    `${S.state.n_cells.toLocaleString()} cells · ${S.state.levels.length} levels` +
    (S.state.representation ? ` · scaler: ${S.state.representation.scaler}` : "") +
    (S.state.demo ? " · demo fixture" : "");
  renderTree();
  renderLedger();
  if (!S.level && S.state.levels.length) selectLevel(S.state.levels[0].name);
  else if (S.level) await renderLevel();
}

let staleClientShown = false;
function showStaleClient() {
  if (staleClientShown) return;
  staleClientShown = true;
  const bar = el("div", {
    style: "position:fixed;left:0;right:0;bottom:0;z-index:99;padding:10px 16px;" +
           "background:var(--warn);color:#12141a;font-weight:600;display:flex;" +
           "align-items:center;gap:12px",
  },
    "This page is running an older build of the client than the server is serving. " +
    "Reload to pick up the current one.",
    el("button", { onclick: () => location.reload() }, "Reload"));
  document.body.append(bar);
}

function currentLevel() {
  return S.state?.levels.find(l => l.name === S.level) || null;
}

function renderTree() {
  const box = $("#levelTree");
  box.innerHTML = "";
  const byParent = new Map();
  for (const lv of S.state.levels) {
    const key = lv.parent || "";
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(lv);
  }
  const walk = (parent, depth) => {
    for (const lv of byParent.get(parent) || []) {
      box.append(el("div", {
        class: "node" + (lv.name === S.level ? " on" : ""),
        style: `padding-left:${7 + depth * 12}px`,
        onclick: () => selectLevel(lv.name),
      },
        el("span", { class: `dot ${lv.status}`, title: lv.status }),
        el("span", { class: "nm" }, lv.name),
        el("span", { class: "ct" }, lv.n_cells ? lv.n_cells.toLocaleString() : ""),
      ));
      walk(lv.name, depth + 1);
    }
  };
  walk("", 0);
}

async function renderLedger() {
  const { entries } = await api("/api/ledger");
  const box = $("#ledger");
  box.innerHTML = "";
  for (const e of entries.slice().reverse()) {
    box.append(el("div", { class: "entry" },
      el("span", { class: "k" }, e.kind), " ",
      el("span", {}, e.level),
      el("div", { class: "d" }, e.detail || ""),
    ));
  }
}

async function selectLevel(name) {
  S.level = name;
  S.cluster = null;
  S.scan = S.preview = S.support = null;
  S.colors = {}; S.names = {};
  resetViews();
  $("#clusterList").innerHTML = "";
  renderTree();
  await renderLevel();
}

async function renderLevel() {
  const lv = currentLevel();
  if (!lv) return;
  $("#crumb").textContent = lv.name;
  $("#levelMeta").textContent =
    `mask ${lv.mask ?? "all"} · ${lv.columns ?? "all features"}` +
    (lv.is_restricted ? ` · restricted ${JSON.stringify(lv.restrict_params)}` : "") +
    (lv.n_cells ? ` · ${lv.n_cells.toLocaleString()} cells` : "") +
    (S.state.representation?.spaces?.[lv.name]
      ? ` · space: ${S.state.representation.spaces[lv.name]}` : "");
  S.names = Object.fromEntries(Object.entries(lv.cut.names || {}));
  S.colors = Object.fromEntries(Object.entries(lv.cut.colors || {}));

  $("#cutBar").classList.toggle("hidden", !lv.has_clustering);
  $("#btnRestrict").disabled = !lv.has_clustering;
  $("#btnAxis").disabled = !lv.has_clustering;
  $("#btnSupport").disabled = !lv.has_clustering;
  $("#btnFlow").disabled = !lv.has_clustering;
  $("#btnEmbed").disabled = !lv.has_clustering;
  updateSweepCost();

  if (lv.has_clustering) {
    await loadScan();
    if (lv.cut.distance_threshold !== null) {
      $("#thresh").value = lv.cut.distance_threshold;
      $("#minSize").value = lv.cut.min_cluster_size;
    }
    await applyThreshold();
    await loadCoverage();
    await loadSupport();
  } else {
    $("#clusterList").innerHTML = "";
    $("#inspectorDetail").innerHTML = "";
  }
}

/* ── sweep ──────────────────────────────────────────────────────────────── */

function renderChips() {
  const gt = $("#graphTypes"); gt.innerHTML = "";
  for (const t of ["knn", "snn_jaccard", "umap_fuzzy"]) {
    gt.append(el("span", {
      class: "chip" + (S.graphTypes.has(t) ? " on" : ""),
      onclick: () => { S.graphTypes.has(t) ? S.graphTypes.delete(t) : S.graphTypes.add(t);
                       renderChips(); updateSweepCost(); },
    }, t));
  }
  const nn = $("#neighbors"); nn.innerHTML = "";
  for (const n of [10, 15, 30, 60, 100]) {
    nn.append(el("span", {
      class: "chip" + (S.neighbors.has(n) ? " on" : ""),
      onclick: () => { S.neighbors.has(n) ? S.neighbors.delete(n) : S.neighbors.add(n);
                       renderChips(); updateSweepCost(); },
    }, String(n)));
  }
}

function updateSweepCost() {
  const graphs = S.graphTypes.size * S.neighbors.size;
  const runs = graphs * (+$("#resN").value) * (+$("#nTimes").value);
  const lv = currentLevel();
  const n = lv?.n_cells ?? S.state?.n_cells ?? 0;
  const gb = (n * (n - 1) / 2) * 8 / 1e9;
  $("#sweepCost").innerHTML =
    `<b>${graphs}</b> graph builds · <b>${runs.toLocaleString()}</b> Leiden runs · ` +
    `linkage ≈ <b>${gb.toFixed(1)} GB</b> (${(2 * gb).toFixed(1)} GB peak)`;
}

async function runCluster(confirm = false) {
  const body = {
    graph_type: [...S.graphTypes],
    n_neighbors: [...S.neighbors],
    n_times: +$("#nTimes").value,
    resolution: geomspace(+$("#resMin").value, +$("#resMax").value, +$("#resN").value),
    confirm,
  };
  try {
    await api(`/api/levels/${S.level}/cluster`, { method: "POST", body });
    S.wasBusy = true;
  } catch (err) {
    if (err.status === 409 && err.data?.needs_confirmation) {
      askModal("This will allocate a large distance matrix",
        `${err.data.n_cells.toLocaleString()} cells needs a ~${err.data.gb} GB condensed
         distance matrix, about ${err.data.peak_gb} GB at peak because scipy works on a
         copy. It cannot be made sparse: a missing entry in the consensus means a pair
         that never co-clustered, which is the largest distance, not zero.`,
        () => runCluster(true));
    } else alert(err.message);
  }
}

const geomspace = (a, b, n) => {
  if (n < 2) return [a];
  const out = [], r = Math.log(b / a) / (n - 1);
  for (let i = 0; i < n; i++) out.push(a * Math.exp(r * i));
  return out;
};

async function loadCoverage() {
  try {
    const token = viewToken();
    const cov = await api(`/api/levels/${S.level}/coverage`);
    if (viewToken() !== token) return;
    const bad = !cov.verdict.startsWith("ok");
    $("#coverage").innerHTML = "";
    $("#coverage").append(
      el("div", { class: "verdict " + (bad ? "bad" : "good") }, cov.verdict),
      el("div", { class: "muted" },
        `${cov.n_runs_in_window} of ${cov.n_runs_total} runs · ` +
        `${cov.n_axes_in_window} of ${cov.n_axes_total} graph settings`),
      table(["n_clusters", "n_runs"], cov.realised_counts.slice(0, 40),
            r => [r.n_clusters, r.n_runs]),
    );
  } catch { $("#coverage").innerHTML = ""; }
}

async function loadAxis() {
  const token = viewToken();
  const ax = await api(`/api/levels/${S.level}/axis-stability`);
  if (viewToken() !== token) return;
  const box = $("#axis"); box.innerHTML = "";
  box.append(el("div", { class: "verdict " + (ax.ranking_stable ? "good" : "bad") },
    ax.ranking_stable
      ? `ranking holds at all three probes: ${ax.ranking.join(" > ")}`
      : "ranking flips between probe thresholds — the audit is not telling you much"));
  for (const p of ax.probes) {
    box.append(el("div", { class: "muted", style: "margin-top:6px" },
      `probe ${p.threshold}`));
    box.append(table([ax.by, "n_runs", "median_clusters", "mean_ari", "min_ari"], p.rows,
      r => [r[ax.by], r.n_runs, fmt(r.median_clusters, 1), fmt(r.mean_ari), fmt(r.min_ari)]));
  }
}

/* ── scan ───────────────────────────────────────────────────────────────── */

async function loadScan() {
  const minSize = +$("#minSize").value;
  const level = S.level;
  const scan = await api(`/api/levels/${S.level}/scan?n_points=60&min_cluster_size=${minSize}`);
  if (S.level !== level) return;          // switched cohorts mid-fetch
  S.scan = scan;
  const slider = $("#thresh");
  slider.min = 0; slider.max = S.scan.max_height;
  slider.step = S.scan.max_height / 500;
  if (!slider.value || +slider.value > S.scan.max_height) {
    slider.value = S.scan.plateaus[0]?.midpoint ?? S.scan.max_height / 2;
  }
  drawScan();
  renderPlateaus();
  stamp("#scanChart");
  stamp("#plateaus");
}

function renderPlateaus() {
  const box = $("#plateaus"); box.innerHTML = "";
  if (!S.scan?.plateaus.length) {
    box.append(el("div", { class: "verdict bad" },
      "No plateau in this range: the cluster count changes at every point. " +
      "That is the useful negative result — this cohort has no scale the ensemble " +
      "agrees on, and no threshold here is defensible. Revisit the grain window."));
    return;
  }
  box.append(el("div", { class: "muted" }, "Plateaus, widest first. Click to take the midpoint."));
  box.append(table(
    ["n_clusters", "width", "midpoint", "max_unassigned", "largest", "median"],
    S.scan.plateaus.slice(0, 8),
    r => [r.n_clusters, fmt(r.width), fmt(r.midpoint), r.max_unassigned,
          r.largest_cluster, fmt(r.median_cluster_size, 1)],
    r => { $("#thresh").value = r.midpoint; applyThreshold(); }));
}

function drawScan() {
  const box = $("#scanChart");
  box.innerHTML = "";
  if (!S.scan?.rows.length) return;
  const rows = S.scan.rows;
  const W = Math.max(box.clientWidth || 800, 520), H = 300;
  const m = { l: 48, r: 52, t: 12, b: 28 };
  const xmax = S.scan.max_height;
  const kmax = Math.max(...rows.map(r => r.n_clusters), 1);
  const nmax = Math.max(...rows.map(r => r.n_assigned + r.n_unassigned), 1);
  const X = v => m.l + (v / xmax) * (W - m.l - m.r);
  const Yk = v => H - m.b - (v / kmax) * (H - m.t - m.b);
  const Yn = v => H - m.b - (v / nmax) * (H - m.t - m.b);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

  // plateau bands first, so the lines read on top of them
  for (const p of (S.scan.plateaus || []).slice(0, 6)) {
    if (p.width <= 0) continue;
    svg.append(svgEl("rect", {
      x: X(p.lo), y: m.t, width: Math.max(X(p.hi) - X(p.lo), 1), height: H - m.t - m.b,
      fill: "#6ea8fe", opacity: 0.09,
    }));
  }
  // the support ceiling: the lowest merge nothing in the ensemble backs
  if (S.support?.ceiling != null) {
    svg.append(svgEl("line", {
      x1: X(S.support.ceiling), x2: X(S.support.ceiling), y1: m.t, y2: H - m.b,
      stroke: "#e45756", "stroke-width": 1.5, "stroke-dasharray": "4 3",
    }));
    svg.append(svgEl("text", {
      x: X(S.support.ceiling) + 4, y: m.t + 11, fill: "#e45756", "font-size": 10,
    }, "support ceiling"));
  }
  // axes
  svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "#2b3040" }));
  for (let i = 0; i <= 4; i++) {
    const v = (kmax / 4) * i;
    svg.append(svgEl("text", { x: m.l - 6, y: Yk(v) + 3, fill: "#8b93a7",
      "font-size": 10, "text-anchor": "end" }, Math.round(v)));
  }
  for (let i = 0; i <= 4; i++) {
    const t = (xmax / 4) * i;
    svg.append(svgEl("text", { x: X(t), y: H - m.b + 14, fill: "#8b93a7",
      "font-size": 10, "text-anchor": "middle" }, t.toFixed(2)));
  }
  const line = (accessor, y, color, dash) => {
    const d = rows.map((r, i) =>
      `${i ? "L" : "M"}${X(r.distance_threshold).toFixed(1)} ${y(accessor(r)).toFixed(1)}`).join("");
    svg.append(svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 1.8,
      "stroke-dasharray": dash || "" }));
  };
  line(r => r.n_clusters, Yk, "#6ea8fe");
  line(r => r.n_unassigned, Yn, "#e45756", "3 3");
  line(r => r.largest_cluster, Yn, "#f0a44a", "5 3");

  // current threshold
  const cur = +$("#thresh").value;
  svg.append(svgEl("line", { x1: X(cur), x2: X(cur), y1: m.t, y2: H - m.b,
    stroke: "#dfe3ec", "stroke-width": 1 }));

  svg.append(svgEl("text", { x: m.l, y: m.t + 10, fill: "#6ea8fe", "font-size": 10 },
    "— n_clusters"));
  svg.append(svgEl("text", { x: m.l + 76, y: m.t + 10, fill: "#e45756", "font-size": 10 },
    "-- unassigned"));
  svg.append(svgEl("text", { x: m.l + 168, y: m.t + 10, fill: "#f0a44a", "font-size": 10 },
    "-- largest"));

  svg.addEventListener("click", ev => {
    const rect = svg.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * (W / rect.width);
    const t = ((px - m.l) / (W - m.l - m.r)) * xmax;
    $("#thresh").value = Math.max(0, Math.min(xmax, t));
    applyThreshold();
  });
  svg.style.cursor = "crosshair";
  box.append(svg);
}

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ── the cut ────────────────────────────────────────────────────────────── */

let applyTimer = null;
function scheduleApply() {
  clearTimeout(applyTimer);
  applyTimer = setTimeout(applyThreshold, 60);
}

async function applyThreshold() {
  const lv = currentLevel();
  if (!lv?.has_clustering) return;
  const t = +$("#thresh").value, minSize = +$("#minSize").value;
  S.preview = await api(
    `/api/levels/${S.level}/preview?distance_threshold=${t}&min_cluster_size=${minSize}`);
  const p = S.preview;
  $("#threshOut").textContent =
    `d=${t.toFixed(3)} · co-cluster ≥ ${p.coclustering_frequency.toFixed(3)}`;
  const v = $("#cutVerdict");
  const parts = [`${p.n_clusters} clusters`, `${p.n_unassigned.toLocaleString()} unassigned`,
                 `largest ${p.largest_cluster.toLocaleString()}`];
  v.textContent = parts.join(" · ");
  v.className = "verdict";
  if (p.above_support_ceiling === true) {
    v.className = "verdict bad";
    v.textContent += " · above the support ceiling — merges here are unsupported";
  }
  drawScan();
  stamp("#scanChart");
  stamp("#plateaus");
  stamp("#support");            // support is threshold-independent; it stays current
  renderClusters();
  renderGradClusters();
  markStale();
  refreshActiveTab();
  const tab = document.querySelector("#tabs button.on")?.dataset.tab;
  if (tab === "features") loadProfile().catch(() => {});
}

function renderClusters() {
  const box = $("#clusterList"); box.innerHTML = "";
  if (!S.preview) return;
  for (const c of S.preview.clusters) {
    const input = el("input", {
      type: "text", value: S.names[c.id] ?? "",
      placeholder: `cluster ${c.id}`,
      onchange: ev => { S.names[c.id] = ev.target.value; pushNames(); },
      onclick: ev => ev.stopPropagation(),
    });
    box.append(el("div", {
      class: "cl" + (S.cluster === c.id ? " on" : ""),
      onclick: () => selectCluster(c.id),
    },
      el("span", { class: "sw", style:
        `background:${colorFor(c.id)};` +
        (S.cluster !== null && S.cluster !== c.id ? "opacity:.35" : "") }),
      input,
      el("span", { class: "n" }, c.n_cells.toLocaleString()),
    ));
  }
  renderCommitPanel();
}

async function pushNames() {
  await api(`/api/levels/${S.level}/names`, {
    method: "POST",
    body: { names: Object.fromEntries(
      Object.entries(S.names).filter(([, v]) => v && v.trim())) },
  });
  renderLedger();
  renderCommitPanel();
}

async function selectCluster(id) {
  S.cluster = S.cluster === id ? null : id;   // click again to clear
  renderClusters();
  drawMap();
  if (S.cluster === null) { $("#inspectorDetail").innerHTML = ""; return; }
  id = S.cluster;
  const box = $("#inspectorDetail"); box.innerHTML = "";
  const c = S.preview.clusters.find(x => x.id === id);
  box.append(el("h2", { style: "margin-top:14px" }, `Cluster ${id}`));
  box.append(el("div", { class: "muted" }, `${c.n_cells.toLocaleString()} cells`));

  if (S.state.neuroglancer.available) {
    const ngl = await api(`/api/levels/${S.level}/clusters/${id}/neuroglancer`);
    box.append(ngl.available
      ? el("div", { class: "row" },
          el("a", { href: ngl.url, target: "_blank", rel: "noopener" },
            el("button", {}, "Open in neuroglancer")),
          el("span", { class: "muted" },
            ngl.truncated ? `${ngl.n_shown} of ${ngl.n_segments} segments` : ""))
      : el("div", { class: "muted" }, ngl.reason));
  }
}

/* ── merge support ──────────────────────────────────────────────────────── */

async function loadSupport() {
  const token = viewToken();
  try {
    const support = await api(`/api/levels/${S.level}/merge-support`);
    if (viewToken() !== token) return;
    S.support = support;
  } catch { S.support = null; }
  const box = $("#support"); box.innerHTML = "";
  stamp("#support");
  if (!S.support?.available) {
    box.append(el("div", { class: "muted" },
      "Not computed for this level yet — it walks every merge against every run."));
    return;
  }
  box.append(el("div", { class: "verdict " + (S.support.ceiling == null ? "good" : "") },
    S.support.ceiling == null
      ? "Every annotated merge is supported by at least one resolution band."
      : `Support ceiling at height ${S.support.ceiling.toFixed(3)} — the lowest merge no
         band backs. A threshold above this is merging groups nothing in the ensemble
         put together.`));
  const bands = S.support.bands || [];
  box.append(table(
    ["height", "freq", "n_cells", ...bands.map((_, i) => `band ${i}`), "verdict"],
    S.support.rows.slice(0, 60),
    r => [fmt(r.height), fmt(r.coclustering_frequency), r.n_cells,
          ...bands.map(b => fmt(r[b], 2)),
          el("span", { class: "tag " + r.verdict.replace("-", "") }, r.verdict)],
    r => { $("#thresh").value = r.height; applyThreshold(); }));
  drawScan();
}

/* ── flow ───────────────────────────────────────────────────────────────── */

async function loadFlow() {
  const n = +$("#flowLevels").value, minSize = +$("#minSize").value;
  const token = viewToken();
  const flow = await api(
    `/api/levels/${S.level}/flow?n_levels=${n}&min_cluster_size=${minSize}`);
  if (viewToken() !== token) return;
  drawFlow(flow);
  stamp("#flowChart");
}

function drawFlow(flow) {
  const box = $("#flowChart"); box.innerHTML = "";
  const W = Math.max(box.clientWidth || 800, 560);
  const H = 420, m = { t: 26, b: 20, l: 10, r: 10 };
  const nLevels = flow.heights.length;
  const colW = (W - m.l - m.r) / nLevels;
  const barW = Math.min(26, colW * 0.28);
  const total = flow.nodes.filter(n => n.level === 0).reduce((a, n) => a + n.n_cells, 0);
  const scale = (H - m.t - m.b) / Math.max(total, 1);

  // Lay out each column: real clusters by size, unassigned pinned to the bottom
  // so the band that drains is always in the same place.
  const pos = new Map();
  for (let L = 0; L < nLevels; L++) {
    const nodes = flow.nodes.filter(n => n.level === L)
      .sort((a, b) => (a.unassigned - b.unassigned) || (b.n_cells - a.n_cells));
    let y = m.t;
    for (const n of nodes) {
      const h = n.n_cells * scale;
      pos.set(`${L}:${n.cluster}`, { x: m.l + L * colW + colW / 2 - barW / 2, y, h, node: n });
      y += h + 2;
    }
  }

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

  // ribbons, thickest first so thin flows stay visible on top
  const cursorL = new Map(), cursorR = new Map();
  for (const e of flow.edges.slice().sort((a, b) => b.value - a.value)) {
    const a = pos.get(`${e.level}:${e.source}`);
    const b = pos.get(`${e.level + 1}:${e.target}`);
    if (!a || !b) continue;
    const kl = `${e.level}:${e.source}`, kr = `${e.level + 1}:${e.target}`;
    const oa = cursorL.get(kl) || 0, ob = cursorR.get(kr) || 0;
    const h = e.value * scale;
    const y0 = a.y + oa, y1 = b.y + ob;
    cursorL.set(kl, oa + h); cursorR.set(kr, ob + h);
    const x0 = a.x + barW, x1 = b.x, mx = (x0 + x1) / 2;
    const d = `M${x0},${y0} C${mx},${y0} ${mx},${y1} ${x1},${y1}` +
              `L${x1},${y1 + h} C${mx},${y1 + h} ${mx},${y0 + h} ${x0},${y0 + h}Z`;
    const unassigned = e.source < 0 || e.target < 0;
    svg.append(svgEl("path", {
      d, fill: unassigned ? "#4a4f5e" : colorFor(e.source),
      opacity: unassigned ? 0.3 : 0.42,
    }, ));
  }
  // nodes
  for (const [, p] of pos) {
    svg.append(svgEl("rect", {
      x: p.x, y: p.y, width: barW, height: Math.max(p.h, 1), rx: 2,
      fill: p.node.unassigned ? "#4a4f5e" : colorFor(p.node.cluster),
    }));
    if (p.h > 12) {
      svg.append(svgEl("text", {
        x: p.x + barW / 2, y: p.y + p.h / 2 + 3, fill: "#12141a",
        "font-size": 9.5, "text-anchor": "middle", "font-weight": 600,
      }, p.node.unassigned ? "—" : p.node.n_cells));
    }
  }
  // column headers: what cut each level is
  flow.heights.forEach((h, i) => {
    svg.append(svgEl("text", {
      x: m.l + i * colW + colW / 2, y: 12, fill: "#8b93a7",
      "font-size": 10, "text-anchor": "middle",
    }, `d=${h.toFixed(2)}`));
    svg.append(svgEl("text", {
      x: m.l + i * colW + colW / 2, y: 22, fill: "#5c6478",
      "font-size": 9, "text-anchor": "middle",
    }, `≥${(flow.max_value - h).toFixed(2)} co-cluster`));
  });
  box.append(svg);
}

/* ── map ────────────────────────────────────────────────────────────────── */

let mapData = null;

async function loadMap() {
  const mode = $$("input[name=colorBy]").find(r => r.checked).value;
  const t = +$("#thresh").value, minSize = +$("#minSize").value;
  const token = viewToken();
  const data = await api(
    `/api/levels/${S.level}/scatter?color_by=${mode}` +
    `&distance_threshold=${t}&min_cluster_size=${minSize}`);
  if (viewToken() !== token) return;
  mapData = data;
  drawMap();
  stamp("#mapCanvas");
}

/* Redrawing is cheap and needs no round trip, so isolating a cluster is
   instant. That interaction is not a nicety: no categorical palette of this
   size passes an all-pairs colourblind check — beyond about eight categories
   colour alone cannot carry identity — so dimming the rest is what actually
   makes a 15-cluster map readable. */
function drawMap() {
  const cv = $("#mapCanvas"), ctx = cv.getContext("2d");
  fitCanvas(cv);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--plot").trim() || "#0d0f14";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!mapData) return;
  if (!mapData.available) {
    ctx.fillStyle = "#8b93a7"; ctx.font = "13px system-ui";
    ctx.fillText(mapData.reason, 16, 26);
    return;
  }
  const data = mapData, n = data.x.length;
  const pad = 20;
  const xr = extent(data.x), yr = extent(data.y);
  const X = v => pad + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * (cv.width - 2 * pad);
  const Y = v => cv.height - pad - ((v - yr[0]) / (yr[1] - yr[0] || 1)) * (cv.height - 2 * pad);

  // Size and alpha both track n. At 35k points, opaque 2.4px squares are a
  // solid mass: the interior of every cluster saturates and all shape
  // information is lost, which is what makes a big UMAP look like poster paint.
  const size = n > 60000 ? 1.2 : n > 20000 ? 1.6 : n > 5000 ? 2.2 : 3.0;
  const alpha = n > 60000 ? 0.30 : n > 20000 ? 0.40 : n > 5000 ? 0.55 : 0.8;
  const half = size / 2;
  const isolating = S.cluster !== null && data.color_by === "cluster";

  // Draw in a fixed shuffled order rather than row order. Row order means the
  // last cluster sits on top of every other one everywhere they overlap, which
  // silently overstates it and hides whatever it covers.
  const order = shuffledOrder(n);
  ctx.globalAlpha = alpha;

  if (isolating) {  // background pass: everything else, grey and faint
    ctx.fillStyle = UNASSIGNED;
    ctx.globalAlpha = alpha * 0.5;
    for (const i of order) {
      if (data.cluster[i] === S.cluster) continue;
      ctx.fillRect(X(data.x[i]) - half, Y(data.y[i]) - half, size, size);
    }
    ctx.globalAlpha = Math.min(1, alpha * 2.2);
  }

  let current = null;
  for (const i of order) {
    if (isolating && data.cluster[i] !== S.cluster) continue;
    const color = data.color_by === "stability"
      ? ramp(data.value[i])
      : colorFor(data.cluster[i]);
    if (color !== current) { ctx.fillStyle = color; current = color; }
    ctx.fillRect(X(data.x[i]) - half, Y(data.y[i]) - half, size, size);
  }
  ctx.globalAlpha = 1;

  drawMapLegend(ctx, cv, data, isolating);
  $("#mapNote").style.opacity = data.color_by === "stability" ? 1 : 0.45;
}

function drawMapLegend(ctx, cv, data, isolating) {
  ctx.font = "11px ui-monospace, Menlo, monospace";
  const lines = [];
  if (data.color_by === "stability") {
    lines.push(["#dfe3ec", "consensus stability: dark = low"]);
  } else if (isolating) {
    const name = S.names[S.cluster] || `cluster ${S.cluster}`;
    lines.push([colorFor(S.cluster), `isolated: ${name}  (click again to clear)`]);
  } else {
    lines.push(["#8b93a7", `${data.x.length.toLocaleString()} cells — click a cluster to isolate`]);
  }
  lines.forEach(([color, text], i) => {
    ctx.fillStyle = color;
    ctx.fillText(text, 12, 18 + i * 15);
  });
}

/* One shuffle per length, cached: a redraw must not resample the draw order or
   the map would shimmer every time you touch the threshold. */
const _orders = new Map();
function shuffledOrder(n) {
  if (_orders.has(n)) return _orders.get(n);
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  let seed = 0x9e3779b9;                      // fixed seed: stable across redraws
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  _orders.set(n, a);
  return a;
}

function fitCanvas(cv) {
  const rect = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(rect.width || cv.clientWidth || 900, 320);
  const h = Math.round(w * 0.62);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv.style.height = `${h}px`;
  }
}

const extent = a => [Math.min(...a), Math.max(...a)];
const ramp = v => {  // dark -> bright; low stability reads as dark
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(20 + 235 * t ** 1.4);
  const g = Math.round(14 + 190 * t ** 1.1);
  const b = Math.round(60 + 90 * (1 - t));
  return `rgb(${r},${g},${b})`;
};

/* ── commit ─────────────────────────────────────────────────────────────── */

function renderCommitPanel() {
  const box = $("#commitClusters"); if (!box) return;
  box.innerHTML = "";
  if (!S.preview) return;
  const named = S.preview.clusters.map(c => ({ ...c, name: S.names[c.id] || String(c.id) }));
  box.append(el("div", { class: "muted" },
    `Committing ${S.preview.n_clusters} clusters at d=${S.preview.distance_threshold.toFixed(3)}, ` +
    `min size ${S.preview.min_cluster_size}. ${S.preview.n_unassigned.toLocaleString()} cells stay unassigned.`));

  // The child builder is rebuilt only when the set of cluster names actually
  // changes. Scrubbing the threshold re-renders this panel constantly, and
  // wiping half-typed child masks on every slider tick would make the last step
  // of the loop unusable.
  const builder = $("#childBuilder");
  const signature = named.map(c => c.name).join("\u0000");
  if (builder.dataset.signature === signature) return;
  builder.dataset.signature = signature;
  builder.innerHTML = "";
  builder.append(el("div", { class: "muted", style: "margin:10px 0 4px" },
    "Group clusters into child masks to descend into. Leave blank to stop here."));
  const rows = el("div", {});
  const addRow = (name = "", members = []) => {
    const sel = el("div", { class: "chips" });
    const chosen = new Set(members);
    for (const c of named) {
      sel.append(el("span", {
        class: "chip" + (chosen.has(c.name) ? " on" : ""),
        onclick: ev => {
          chosen.has(c.name) ? chosen.delete(c.name) : chosen.add(c.name);
          ev.target.classList.toggle("on");
          row.dataset.members = JSON.stringify([...chosen]);
        },
      }, c.name));
    }
    const row = el("div", { class: "row", style: "align-items:flex-start" },
      el("input", { type: "text", placeholder: "child mask name", class: "childname" }),
      sel);
    row.dataset.members = JSON.stringify([...chosen]);
    rows.append(row);
  };
  addRow();
  builder.append(rows,
    el("button", { onclick: () => addRow() }, "+ another child"));
}

async function commitLevel() {
  const children = {};
  for (const row of $$("#childBuilder .row")) {
    const name = row.querySelector(".childname")?.value?.trim();
    const members = JSON.parse(row.dataset.members || "[]");
    if (name && members.length) children[name] = members;
  }
  const body = { label_name: $("#labelName").value.trim() || null, children };
  try {
    const res = await api(`/api/levels/${S.level}/commit`, { method: "POST", body });
    await refreshAll();
    if (res.children.length) selectLevel(res.children[0]);
  } catch (err) { alert(err.message); }
}

/* ── small helpers ──────────────────────────────────────────────────────── */

function table(headers, rows, cells, onPick) {
  const t = el("table", {});
  t.append(el("thead", {}, el("tr", {}, headers.map(h =>
    el("th", { class: typeof h === "string" && h !== headers[0] ? "num" : "" }, h)))));
  const body = el("tbody", {});
  for (const r of rows) {
    const tr = el("tr", onPick ? { class: "pick", onclick: () => onPick(r) } : {});
    cells(r).forEach((c, i) => tr.append(el("td", { class: i ? "num" : "" }, c)));
    body.append(tr);
  }
  t.append(body);
  return t;
}

function askModal(title, body, onOk) {
  $("#modalTitle").textContent = title;
  $("#modalBody").textContent = body;
  const dlg = $("#modal");
  $("#modalOk").onclick = () => { dlg.close(); onOk(); };
  $("#modalCancel").onclick = () => dlg.close();
  dlg.showModal();
}

/* ── wiring ─────────────────────────────────────────────────────────────── */

$("#tabs").addEventListener("click", ev => {
  const btn = ev.target.closest("button[data-tab]");
  if (!btn) return;
  $$("#tabs button").forEach(b => b.classList.toggle("on", b === btn));
  $$(".panel").forEach(p => p.classList.toggle("on", p.dataset.panel === btn.dataset.tab));
  if (btn.dataset.tab === "scan") drawScan();
  if (btn.dataset.tab === "map") loadMap();
  if (btn.dataset.tab === "features") {
    loadProfile().catch(() => {});
    loadHeatmap().catch(() => {});
  }
  if (btn.dataset.tab === "continuum") {
    loadColumns().catch(() => {});
    renderGradClusters();
    loadBoundary().catch(() => {});
    loadGradient().catch(() => {});
  }
});

$("#thresh").addEventListener("input", scheduleApply);
$("#minSize").addEventListener("change", async () => { await loadScan(); applyThreshold(); });
$("#btnCluster").addEventListener("click", () => runCluster(false));
async function runRestrict(confirm = false) {
  const body = { confirm };
  if ($("#ncMin").value) body.n_clusters_min = +$("#ncMin").value;
  if ($("#ncMax").value) body.n_clusters_max = +$("#ncMax").value;
  try {
    await api(`/api/levels/${S.level}/restrict`, { method: "POST", body });
    S.wasBusy = true;
  } catch (err) {
    if (err.status === 409 && err.data?.kind === "dense_consensus") {
      askModal("This window makes the consensus matrix denser, not sparser",
        `Projected ${(err.data.density * 100).toFixed(0)}% dense over ` +
        `${err.data.n_cells.toLocaleString()} cells: about ${err.data.gb} GB for the ` +
        `matrix and ${err.data.peak_gb} GB at peak. Narrowing to a coarse grain window ` +
        `selects the runs with the biggest clusters, and A@A.T gains a size^2 block per ` +
        `cluster — so coarser windows cost more memory, not less. Widen the window, or ` +
        `raise n_clusters_min, to come back down.`,
        () => runRestrict(true));
    } else alert(err.message);
  }
}
$("#btnRestrict").addEventListener("click", () => runRestrict(false));
$("#btnAxis").addEventListener("click", () => loadAxis().catch(e => alert(e.message)));
$("#btnSupport").addEventListener("click", async () => {
  await api(`/api/levels/${S.level}/merge-support?n_bands=${+$("#nBands").value}`,
            { method: "POST" });
  S.wasBusy = true;
});
$("#btnEmbed").addEventListener("click", async () => {
  await api(`/api/levels/${S.level}/embed`,
            { method: "POST", body: { method: $("#embedMethod").value } });
  S.wasBusy = true;
});
$("#btnFlow").addEventListener("click", () => loadFlow().catch(e => alert(e.message)));
$("#btnMap").addEventListener("click", () => loadMap().catch(e => alert(e.message)));
$("#btnCommit").addEventListener("click", commitLevel);
/* Reset offers the two scopes explicitly rather than picking one, because they
   cost wildly different amounts: keeping the sweeps is instant, dropping them
   means re-running every linkage. Neither is reachable without a click here —
   the route refuses an unconfirmed request. */
$("#btnReset").addEventListener("click", () => {
  const entries = $("#ledger").childElementCount;
  $("#modalTitle").textContent = "Reset the descent";
  const body = $("#modalBody");
  body.innerHTML = "";
  body.append(el("p", { class: "note" },
    `${entries} ledger entries. Resetting also undoes what they wrote — it detaches ` +
    `the label columns and drops the child masks, so the ledger and the table stay ` +
    `in step. The current ledger is copied to a .bak file first.`));
  body.append(el("div", { class: "choice", onclick: () => doReset(true) },
    el("b", {}, "Reset decisions, keep the sweeps"),
    el("span", {}, "Drops cuts, names, commits, boundaries and gradients. Keeps the " +
      "consensus sweeps and grain windows — so you can re-cut immediately, with no " +
      "linkage to rebuild.")));
  body.append(el("div", { class: "choice", onclick: () => doReset(false) },
    el("b", {}, "Reset everything"),
    el("span", {}, "Clean slate. Every sweep has to be re-run, which at this cohort " +
      "size means paying for the linkage again.")));
  $("#modalOk").style.display = "none";
  $("#modalCancel").onclick = () => { $("#modal").close(); $("#modalOk").style.display = ""; };
  $("#modal").showModal();
});

async function doReset(keepSweeps) {
  $("#modal").close();
  $("#modalOk").style.display = "";
  try {
    const res = await api("/api/ledger/rewind", {
      method: "POST", body: { confirm: true, keep_sweeps: keepSweeps },
    });
    S.level = null; S.cluster = null;
    S.scan = S.preview = S.support = null;
    S.colors = {}; S.names = {};
    resetViews();
    $("#clusterList").innerHTML = "";
    await refreshAll();
    const undone = res.undone?.length ? "\n\n" + res.undone.join("\n") : "";
    alert(`Dropped ${res.dropped} entries, ${res.remaining} remain.` + undone);
  } catch (err) { alert(err.message); }
}

$("#btnScript").addEventListener("click", async () => {
  const text = await (await fetch("/api/ledger/script")).text();
  askModal("The descent as a standalone script", text, () => {});
  $("#modalBody").innerHTML = "";
  $("#modalBody").append(el("pre", {
    style: "max-height:50vh;overflow:auto;font-size:11px;white-space:pre-wrap",
  }, text));
});
for (const id of ["#resN", "#nTimes"]) $(id).addEventListener("input", updateSweepCost);
window.addEventListener("resize", () => { drawScan(); drawMap(); });


/* ── features: what the clusters are made of ────────────────────────────── */

let columnsCache = null;

async function loadColumns() {
  if (columnsCache && columnsCache.level === S.level) return columnsCache;
  const data = await api(`/api/levels/${S.level}/columns`);
  columnsCache = { level: S.level, ...data };
  const numeric = data.numeric.map(c => c.name);
  fillSelect("#profileColumn", numeric, currentLevel()?.order_by);
  fillSelect("#fsAgainst", numeric, currentLevel()?.order_by);
  fillSelect("#fsFeature", data.features);
  fillSelect("#gradOrient", ["", ...numeric], currentLevel()?.order_by);
  fillSelect("#gradNuisance", numeric);
  return columnsCache;
}

function fillSelect(sel, options, preferred) {
  const node = $(sel);
  if (!node || node.dataset.filled === String(options.length) + S.level) return;
  const keep = node.value;
  node.innerHTML = "";
  for (const o of options) node.append(el("option", { value: o }, o || "—"));
  node.value = options.includes(keep) ? keep
             : (preferred && options.includes(preferred) ? preferred : options[0] || "");
  node.dataset.filled = String(options.length) + S.level;
}

async function loadProfile() {
  await loadColumns();
  const column = $("#profileColumn").value;
  if (!column) return;
  const token = viewToken();
  const data = await api(
    `/api/levels/${S.level}/profile?column=${encodeURIComponent(column)}` +
    `&bins=${+$("#profileBins").value}`);
  if (viewToken() !== token) return;
  drawProfile(data, $("#profileNorm").checked);
  stamp("#profileChart");
}

/* A ridgeline rather than stacked bars: the question is where each cluster sits
   relative to the others, and stacking hides exactly that by making every
   cluster's baseline depend on the ones drawn under it. */
function drawProfile(data, normalize) {
  const box = $("#profileChart"); box.innerHTML = "";
  const rows = data.series;
  if (!rows.length) return;
  const W = Math.max(box.clientWidth || 800, 520);
  const rowH = 46, m = { l: 132, r: 20, t: 14, b: 30 };
  const H = m.t + m.b + rows.length * rowH;
  const X = v => m.l + ((v - data.lo) / (data.hi - data.lo || 1)) * (W - m.l - m.r);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

  const peak = normalize
    ? Math.max(...rows.map(r => Math.max(...r.fraction)))
    : Math.max(...rows.map(r => Math.max(...r.counts)));

  rows.forEach((row, i) => {
    const base = m.t + i * rowH + rowH - 8;
    const values = normalize ? row.fraction : row.counts;
    const pts = values.map((v, j) => {
      const x = X(data.centers[j]);
      const y = base - (v / (peak || 1)) * (rowH - 14);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    svg.append(svgEl("path", {
      d: `M${X(data.lo)},${base} L${pts.join(" L")} L${X(data.hi)},${base}Z`,
      fill: colorFor(row.cluster), opacity: 0.55,
      stroke: colorFor(row.cluster), "stroke-width": 1.2,
    }));
    // the quartile bar: shape can be noisy, position rarely is
    if (row.q1 !== null) {
      svg.append(svgEl("line", {
        x1: X(row.q1), x2: X(row.q3), y1: base + 4, y2: base + 4,
        stroke: colorFor(row.cluster), "stroke-width": 2, opacity: 0.9,
      }));
      svg.append(svgEl("circle", {
        cx: X(row.median), cy: base + 4, r: 2.6, fill: "#dfe3ec",
      }));
    }
    svg.append(svgEl("text", {
      x: m.l - 8, y: base - 4, fill: "#dfe3ec", "font-size": 11, "text-anchor": "end",
    }, row.name));
    svg.append(svgEl("text", {
      x: m.l - 8, y: base + 8, fill: "#8b93a7", "font-size": 9.5, "text-anchor": "end",
    }, `${row.n_cells.toLocaleString()} cells`));
  });
  for (let i = 0; i <= 4; i++) {
    const v = data.lo + (data.hi - data.lo) * (i / 4);
    svg.append(svgEl("text", {
      x: X(v), y: H - 10, fill: "#8b93a7", "font-size": 10, "text-anchor": "middle",
    }, v.toFixed(v > 100 ? 0 : 2)));
  }
  svg.append(svgEl("text", {
    x: (W + m.l) / 2, y: H - 22, fill: "#5c6478", "font-size": 10, "text-anchor": "middle",
  }, data.column));
  box.append(svg);
}

async function loadHeatmap() {
  const token = viewToken();
  const data = await api(`/api/levels/${S.level}/heatmap?top=${+$("#heatTop").value}`);
  if (viewToken() !== token) return;
  const box = $("#heatmapChart"); box.innerHTML = "";
  stamp("#heatmapChart");
  if (!data.features.length) {
    box.append(el("div", { class: "muted" }, data.reason || "nothing to show"));
    return;
  }
  const cellW = 22, cellH = 22, m = { l: 132, t: 96, r: 16, b: 12 };
  const W = m.l + m.r + data.features.length * cellW;
  const H = m.t + m.b + data.clusters.length * cellH;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  const vmax = data.vmax || 1;

  data.matrix.forEach((row, r) => {
    row.forEach((v, c) => {
      // Build the cell, hang its tooltip on it, then append. Node.append()
      // returns undefined, so it cannot be chained.
      const cell = svgEl("rect", {
        x: m.l + c * cellW, y: m.t + r * cellH,
        width: cellW - 2, height: cellH - 2, rx: 2,
        fill: diverging(v / vmax),
      });
      cell.append(svgEl("title", {},
        `${data.clusters[r].name} · ${data.features[c]}: ${v.toFixed(2)} sd`));
      svg.append(cell);
    });
    svg.append(svgEl("text", {
      x: m.l - 8, y: m.t + r * cellH + 15, fill: "#dfe3ec",
      "font-size": 11, "text-anchor": "end",
    }, data.clusters[r].name));
    svg.append(svgEl("rect", {
      x: m.l - 4, y: m.t + r * cellH + 4, width: 3, height: cellH - 10,
      fill: colorFor(data.clusters[r].cluster),
    }));
  });
  data.features.forEach((f, c) => {
    svg.append(svgEl("text", {
      x: m.l + c * cellW + cellW / 2 - 3, y: m.t - 8, fill: "#8b93a7", "font-size": 10,
      transform: `rotate(-60 ${m.l + c * cellW + cellW / 2 - 3} ${m.t - 8})`,
    }, f));
  });
  svg.append(svgEl("text", { x: m.l, y: 14, fill: "#5c6478", "font-size": 10 },
    "z-scored cluster means, most discriminative first — blue below cohort mean, orange above"));
  box.append(svg);
}

/* Diverging blue↔orange with a neutral midpoint: this encodes polarity (below
   or above the cohort mean), so a single-hue ramp would hide the sign and a
   rainbow would invent structure at the middle. */
const diverging = t => {
  const v = Math.max(-1, Math.min(1, t));
  const mix = (a, b, k) => a.map((x, i) => Math.round(x + (b[i] - x) * k));
  const mid = [30, 34, 44], lo = [31, 119, 180], hi = [255, 127, 14];
  const c = v < 0 ? mix(mid, lo, -v) : mix(mid, hi, v);
  return `rgb(${c[0]},${c[1]},${c[2]})`;
};

async function loadFeatureScatter() {
  await loadColumns();
  const feature = $("#fsFeature").value, against = $("#fsAgainst").value;
  if (!feature || !against) return;
  const token = viewToken();
  const data = await api(
    `/api/levels/${S.level}/feature-scatter?feature=${encodeURIComponent(feature)}` +
    `&against=${encodeURIComponent(against)}&scaled=${$("#fsScaled").checked}`);
  if (viewToken() !== token) return;
  stamp("#fsCanvas");
  const cv = $("#fsCanvas"), ctx = cv.getContext("2d");
  fitCanvas(cv);
  ctx.fillStyle = "#0d0f14"; ctx.fillRect(0, 0, cv.width, cv.height);
  const pad = { l: 54, r: 14, t: 14, b: 34 };
  if (!data.x.length) return;
  const xr = extent(data.x), yr = extent(data.y);
  const X = v => pad.l + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * (cv.width - pad.l - pad.r);
  const Y = v => cv.height - pad.b - ((v - yr[0]) / (yr[1] - yr[0] || 1)) * (cv.height - pad.t - pad.b);
  const n = data.x.length;
  const size = n > 20000 ? 1.6 : n > 5000 ? 2.2 : 3.0;
  ctx.globalAlpha = n > 20000 ? 0.4 : 0.65;
  const isolating = S.cluster !== null;
  for (const i of shuffledOrder(n)) {
    const dim = isolating && data.cluster[i] !== S.cluster;
    ctx.fillStyle = dim ? UNASSIGNED : colorFor(data.cluster[i]);
    ctx.globalAlpha = dim ? 0.15 : (n > 20000 ? 0.45 : 0.7);
    ctx.fillRect(X(data.x[i]) - size / 2, Y(data.y[i]) - size / 2, size, size);
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#8b93a7"; ctx.font = "11px ui-monospace, Menlo, monospace";
  ctx.fillText(data.against, cv.width / 2 - 30, cv.height - 10);
  ctx.save(); ctx.translate(14, cv.height / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(data.feature + (data.scaled ? " (scaled)" : ""), -30, 0); ctx.restore();
  [xr, yr].forEach((r, axis) => {
    for (let i = 0; i <= 3; i++) {
      const v = r[0] + (r[1] - r[0]) * (i / 3);
      const txt = Math.abs(v) > 100 ? v.toFixed(0) : v.toFixed(2);
      if (axis === 0) ctx.fillText(txt, X(v) - 12, cv.height - pad.b + 15);
      else ctx.fillText(txt, 24, Y(v) + 4);
    }
  });
}

/* ── continuum: gap or cut, and the coordinate if it is a cut ───────────── */

async function loadBoundary() {
  const token = viewToken();
  const data = await api(`/api/levels/${S.level}/boundary`);
  if (viewToken() !== token) return;
  const box = $("#boundary"); box.innerHTML = "";
  stamp("#boundary");
  if (!data.available) {
    box.append(el("div", { class: "muted" },
      "Not run for this cut yet. It rebuilds the space, builds a kNN graph and " +
      "runs a dip test per pair, so it is a job rather than a live read."));
    return;
  }
  const counts = data.counts || {};
  box.append(el("div", { class: "row" }, Object.entries(counts).map(([k, v]) =>
    el("span", { class: "tag " + (k === "continuous" ? "fineonly" : k === "discrete" ? "allband" : "unsupported") },
      `${v} ${k}`))));
  if (counts.continuous)
    box.append(el("div", { class: "verdict bad" },
      `${counts.continuous} boundary(ies) look continuous — those pairs are candidates ` +
      `for a coordinate rather than a label. Select them below.`));

  const cols = ["cluster_a", "cluster_b", "verdict", "dip_p", "connectivity_ratio",
                "valley_ratio", "cocluster_cross_mean"];
  const present = cols.filter(c => (data.columns || []).includes(c));
  box.append(table(present.map(c => c.replace(/_/g, " ")), data.rows,
    r => present.map(c =>
      c === "verdict"
        ? el("span", { class: "tag " + (r[c] === "discrete" ? "allband"
            : r[c] === "continuous" ? "fineonly" : "unsupported") }, r[c])
        : fmt(r[c])),
    r => {  // clicking a row preselects that pair for parametrization
      gradPick = new Set([String(r.cluster_a), String(r.cluster_b)]);
      renderGradClusters();
    }));
}

let gradPick = new Set();

function renderGradClusters() {
  const box = $("#gradClusters"); if (!box) return;
  box.innerHTML = "";
  if (!S.preview) return;
  for (const c of S.preview.clusters) {
    const key = S.names[c.id] || String(c.id);
    box.append(el("span", {
      class: "chip" + (gradPick.has(key) || gradPick.has(String(c.id)) ? " on" : ""),
      onclick: () => {
        const k = S.names[c.id] || String(c.id);
        gradPick.has(k) ? gradPick.delete(k) : gradPick.add(k);
        renderGradClusters();
      },
    }, key));
  }
}

async function runParametrize() {
  const clusters = [...gradPick];
  if (clusters.length < 2) { alert("pick at least two clusters"); return; }
  const nuisance = Array.from($("#gradNuisance").selectedOptions).map(o => o.value);
  await api(`/api/levels/${S.level}/parametrize`, {
    method: "POST",
    body: {
      clusters,
      orient_by: $("#gradOrient").value || null,
      nuisance: nuisance.length ? nuisance : null,
    },
  });
  S.wasBusy = true;
}

async function loadGradient() {
  const token = viewToken();
  const g = await api(`/api/levels/${S.level}/gradient`);
  if (viewToken() !== token) return;
  const box = $("#gradient"); box.innerHTML = "";
  stamp("#gradient");
  if (!g.available) {
    box.append(el("div", { class: "muted" }, "No axis fitted for this level."));
    return;
  }
  // Warnings first and unmissable: they are the reason not to keep this axis.
  for (const w of g.warnings || []) {
    box.append(el("div", { class: "verdict bad", style: "margin-bottom:6px" }, "⚠ " + w));
  }
  box.append(el("div", { class: "muted" },
    `${g.name} · ${g.n_cells.toLocaleString()} cells` +
    (g.intrinsic_dimension !== null
      ? ` · intrinsic dimension ${g.intrinsic_dimension.toFixed(2)}` : "")));

  if (g.dimension_profile?.length) drawDimensionProfile(box, g);

  box.append(el("div", { style: "margin-top:8px" },
    el("div", { class: "muted" }, "Features varying along the axis (Spearman ρ)")));
  box.append(table(["feature", "ρ"], g.loadings.slice(0, 15),
    r => [r.feature, fmt(r.spearman_rho)]));

  box.append(el("div", { class: "row", style: "margin-top:10px" },
    el("span", { class: "muted" }, "Commit:"),
    el("button", { onclick: () => commitGradient(null) }, "coordinate only"),
    el("label", {}, "bins ", el("input", { type: "number", id: "gradBins", value: 3, min: 2, max: 12 })),
    el("button", { onclick: () => commitGradient(+$("#gradBins").value) }, "coordinate + declared bins"),
  ));
  box.append(el("p", { class: "note", style: "margin-top:6px" },
    "Attaching the coordinate alone keeps the continuum a continuum. Binning adds " +
    "named intervals alongside it — the coordinate stays attached, so the labels " +
    "stay honest and the cut points are declared numbers anyone can revise."));
}

async function commitGradient(bins) {
  try {
    const res = await api(`/api/levels/${S.level}/gradient/commit`, {
      method: "POST", body: { bins },
    });
    await refreshAll();
    alert(`attached ${res.coordinate}` + (res.bins ? ` and ${res.bins}` : ""));
  } catch (err) { alert(err.message); }
}

/* Refresh only the panel in front of you, and only once the threshold settles.

   Scrubbing fires many previews a second; each of these is a server round trip
   over tens of thousands of rows, so refetching per tick would make the slider
   the slowest control in the app. The stale marking is what makes the delay
   honest in the meantime — the old picture is visibly labelled, not passed off
   as current. Panels on hidden tabs stay stale until you look at them. */
let refreshTimer = null;
function refreshActiveTab() {
  clearTimeout(refreshTimer);
  const tab = document.querySelector("#tabs button.on")?.dataset.tab;
  if (!["map", "features"].includes(tab)) return;
  refreshTimer = setTimeout(() => {
    if (tab === "map") loadMap().catch(() => {});
    if (tab === "features") {
      loadProfile().catch(() => {});
      loadHeatmap().catch(() => {});
    }
  }, 450);
}


/* Intrinsic dimension against decimation scale.

   The reported dimension is the *minimum* over these levels, which is the right
   conservative choice for a gate and throws away the informative part. The
   shape is what says whether a curve was the right model at all:

     flat and low (~1)  a curve at every scale — parametrize is the right tool
     flat and high      genuinely that many dimensions; a 1-D axis is an artifact
     falling with scale noise-dominated up close; the real dimension is the floor
     rising with scale  curvature — locally 1-D, but it folds

   Decimating thins the sample, which *grows* the typical nearest-neighbour
   distance, so left-to-right on this axis is small scale to large. */
function drawDimensionProfile(box, g) {
  const rows = g.dimension_profile;
  const W = 420, H = 150, m = { l: 34, r: 12, t: 22, b: 30 };
  const dims = rows.flatMap(r => [r.dimension + r.spread, r.dimension - r.spread]);
  const top = Math.max(...dims, g.dimension_gate + 0.5, 2);
  const X = i => m.l + (rows.length < 2 ? 0.5 : i / (rows.length - 1)) * (W - m.l - m.r);
  const Y = v => H - m.b - (v / top) * (H - m.t - m.b);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

  // the gate, and the dimension a curve would have
  for (const [value, color, label] of [
    [g.dimension_gate, "#e45756", `gate ${g.dimension_gate}`],
    [1, "#54a24b", "a curve"],
  ]) {
    if (value > top) continue;
    svg.append(svgEl("line", {
      x1: m.l, x2: W - m.r, y1: Y(value), y2: Y(value),
      stroke: color, "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.65,
    }));
    svg.append(svgEl("text", {
      x: W - m.r, y: Y(value) - 3, fill: color, "font-size": 9, "text-anchor": "end",
    }, label));
  }

  rows.forEach((r, i) => {
    if (r.spread > 0) {  // draws disagree — the estimate is not determined here
      svg.append(svgEl("line", {
        x1: X(i), x2: X(i), y1: Y(r.dimension - r.spread), y2: Y(r.dimension + r.spread),
        stroke: "#6ea8fe", "stroke-width": 1, opacity: 0.5,
      }));
    }
    svg.append(svgEl("circle", { cx: X(i), cy: Y(r.dimension), r: 3, fill: "#6ea8fe" }));
    svg.append(svgEl("text", {
      x: X(i), y: H - m.b + 12, fill: "#8b93a7", "font-size": 9, "text-anchor": "middle",
    }, `n=${Math.round(r.n)}`));
  });
  svg.append(svgEl("path", {
    d: rows.map((r, i) => `${i ? "L" : "M"}${X(i)},${Y(r.dimension)}`).join(" "),
    fill: "none", stroke: "#6ea8fe", "stroke-width": 1.6,
  }));
  for (let i = 0; i <= 2; i++) {
    const v = (top / 2) * i;
    svg.append(svgEl("text", {
      x: m.l - 6, y: Y(v) + 3, fill: "#8b93a7", "font-size": 9, "text-anchor": "end",
    }, v.toFixed(0)));
  }
  svg.append(svgEl("text", { x: m.l, y: 12, fill: "#5c6478", "font-size": 10 },
    "intrinsic dimension vs scale (thinner sample = larger scale)"));
  svg.append(svgEl("text", {
    x: (W + m.l) / 2, y: H - 4, fill: "#5c6478", "font-size": 9, "text-anchor": "middle",
  }, verdictFor(rows)));

  box.append(el("div", { style: "margin:8px 0" }, svg));
}

/* Say what the shape means, rather than leaving it to be squinted at. The
   thresholds are deliberately loose — this is a decimation proxy, not GRIDE. */
function verdictFor(rows) {
  if (rows.length < 2) return "one scale only — nothing to compare";
  const first = rows[0].dimension, last = rows[rows.length - 1].dimension;
  const drop = first - last;
  const flat = Math.abs(drop) < 0.5;
  if (flat && last < 1.5) return "flat and low — a curve at every scale";
  if (flat) return `flat near ${last.toFixed(1)} — genuinely that many dimensions`;
  if (drop > 0) return `falls ${first.toFixed(1)} -> ${last.toFixed(1)} — noise-dominated up close`;
  return `rises ${first.toFixed(1)} -> ${last.toFixed(1)} — curvature, locally 1-D but folding`;
}

/* ── wiring for the new tabs ────────────────────────────────────────────── */

$("#profileColumn").addEventListener("change", () => loadProfile().catch(e => alert(e.message)));
$("#profileBins").addEventListener("change", () => loadProfile().catch(e => alert(e.message)));
$("#profileNorm").addEventListener("change", () => loadProfile().catch(e => alert(e.message)));
$("#btnHeatmap").addEventListener("click", () => loadHeatmap().catch(e => alert(e.message)));
$("#btnFeatureScatter").addEventListener("click", () => loadFeatureScatter().catch(e => alert(e.message)));
$("#btnBoundary").addEventListener("click", async () => {
  await api(`/api/levels/${S.level}/boundary`, {
    method: "POST", body: { n_neighbors: +$("#boundaryK").value } });
  S.wasBusy = true;
});
$("#btnParametrize").addEventListener("click", () => runParametrize().catch(e => alert(e.message)));

/* Bootstrap last, so every handler above is attached first. Polling starts
   whichever way the first load goes — if it starts only on success, a single
   transient failure at startup leaves the page permanently inert with nothing
   on screen to say so.

   Wrapped in one named function and invoked on one line so the render-test
   harness can suppress it by removing that line. A multi-line bootstrap once
   left the harness stripping only its first line and syntax-erroring on the
   remainder. */
function bootstrap() {
  renderChips();
  refreshAll()
    .catch(err => {
      const pill = $("#jobPill");
      pill.className = "pill failed";
      pill.textContent = `initial load failed: ${err.message}`;
    })
    .finally(watchJobs);
}

bootstrap();
