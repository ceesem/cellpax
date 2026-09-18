/* Execute every chart renderer against real server payloads, in jsdom.
 *
 * The app ships no build step and no JS test runner, which is the right call for
 * a local tool — but it meant the drawing code was only ever exercised by a
 * human looking at it, and two rendering bugs shipped that way. This closes the
 * gap without adding a framework: load app.js into a DOM built from the real
 * index.html, hand each renderer a fixture captured from the actual API, and
 * fail on any thrown error or unhandled rejection.
 *
 * It checks that the code runs and produces marks. It cannot check that the
 * result looks right — that still needs eyes.
 *
 * Usage:  node tests/js/render_smoke.mjs <fixtures.json>
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const here = dirname(fileURLToPath(import.meta.url));
const staticDir = resolve(here, "../../src/cellpax/app/static");
const fixtures = JSON.parse(readFileSync(process.argv[2], "utf8"));

const html = readFileSync(resolve(staticDir, "index.html"), "utf8");
const source = readFileSync(resolve(staticDir, "app.js"), "utf8");

const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true });
const { window } = dom;

// Canvas has no backend in jsdom; a recording stub is enough to prove the
// drawing code runs and issues the calls it means to.
const calls = { fillRect: 0, fillText: 0, clearRect: 0 };
window.HTMLCanvasElement.prototype.getContext = () => ({
  fillRect: () => calls.fillRect++,
  clearRect: () => calls.clearRect++,
  fillText: () => calls.fillText++,
  save() {}, restore() {}, translate() {}, rotate() {},
  set fillStyle(_v) {}, get fillStyle() { return "#000"; },
  set globalAlpha(_v) {}, get globalAlpha() { return 1; },
  set font(_v) {}, get font() { return ""; },
});

const failures = [];
window.addEventListener("error", e => failures.push(`window error: ${e.message}`));
window.alert = msg => failures.push(`alert(): ${msg}`);

/* Serve the captured payloads instead of a server, so the *real* async loaders
   run — not hand-picked draw functions. That distinction matters: the bug this
   harness exists to catch lived in loadHeatmap, and a test that only called a
   drawing helper would have sailed straight past it. An unmapped URL is a
   failure, not a blank response, so adding an endpoint without a fixture shows
   up here rather than silently going uncovered. */
const ROUTES = {
  "state": "state", "jobs": "jobs", "ledger": "ledger",
  "scan": "scan", "preview": "preview", "profile": "profile",
  "heatmap": "heatmap", "flow": "flow", "scatter": "scatter",
  "feature-scatter": "featureScatter", "columns": "columns",
  "boundary": "boundary", "gradient": "gradient",
  "merge-support": "mergeSupport", "coverage": "coverage",
  "axis-stability": "axisStability", "soft-labels": "softLabels",
};
const served = new Set();
window.fetch = async (url) => {
  const path = String(url).split("?")[0];
  const leaf = path.split("/").filter(Boolean).pop();
  const key = ROUTES[leaf];
  if (key === undefined || fixtures[key] === undefined) {
    failures.push(`no fixture for ${url} (leaf ${leaf!== undefined ? leaf : "?"})`);
    return { ok: false, status: 500, text: async () => "{}" };
  }
  served.add(leaf);
  return { ok: true, status: 200, text: async () => JSON.stringify(fixtures[key]) };
};

// app.js bootstraps itself with refreshAll(), which fetches. Strip that line
// wherever it sits — it is no longer at EOF — and expose the top-level bindings
// so the renderers can be called directly.
const BOOTSTRAP = /^bootstrap\(\);$/m;
if (!BOOTSTRAP.test(source)) {
  throw new Error("bootstrap() call not found — app.js changed shape, update the harness");
}
const body = source.replace(BOOTSTRAP, "// bootstrap() suppressed by the render harness");
const exposed = `
${body}
globalThis.__api = {
  S, drawProfile, drawFlow, drawScan, drawMap, drawFeatureScatterFrom,
  renderPlateaus, table, viewToken, resetViews, markStale, colorFor,
  loadHeatmap, loadProfile, loadBoundary, loadGradient, loadScan, loadFlow,
  loadMap, loadFeatureScatter, loadCoverage, loadAxis, loadSupport,
  applyThreshold, refreshAll, watchJobs, renderLevel,
  stopJobs: () => clearTimeout(pollTimer),
  setMapData: d => { mapData = d; },
  setState: s => { S.state = s; },
};
`;

// drawMap reads a module-global; give the harness a way in without exporting it.
const shimmed = exposed.replace(
  "globalThis.__api = {",
  `function drawFeatureScatterFrom(data) {
     const cv = document.querySelector("#fsCanvas"), ctx = cv.getContext("2d");
     fitCanvas(cv);
     const pad = { l: 54, r: 14, t: 14, b: 34 };
     if (!data.x.length) return;
     const xr = extent(data.x), yr = extent(data.y);
     const X = v => pad.l + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * (cv.width - pad.l - pad.r);
     const Y = v => cv.height - pad.b - ((v - yr[0]) / (yr[1] - yr[0] || 1)) * (cv.height - pad.t - pad.b);
     for (const i of shuffledOrder(data.x.length)) {
       ctx.fillStyle = colorFor(data.cluster[i]);
       ctx.fillRect(X(data.x[i]), Y(data.y[i]), 2, 2);
     }
   }
   globalThis.__api = {`
);

window.eval(shimmed);
const api = window.__api;

async function checkAsync(name, fn) {
  try {
    await fn();
    process.stdout.write(`  ok    ${name}\n`);
  } catch (err) {
    failures.push(`${name}: ${err.message}`);
    process.stdout.write(`  FAIL  ${name}: ${err.message}\n`);
  }
}

function check(name, fn) {
  try {
    fn();
    process.stdout.write(`  ok    ${name}\n`);
  } catch (err) {
    failures.push(`${name}: ${err.message}`);
    process.stdout.write(`  FAIL  ${name}: ${err.message}\n`);
  }
}

const $ = sel => window.document.querySelector(sel);
const marks = sel => $(sel).querySelectorAll("rect, path, circle, text").length;

api.setState(fixtures.state);
api.S.level = fixtures.level;
api.S.preview = fixtures.preview;

/* Drive the real client entry points wherever one exists. Calling a drawing
   helper directly proves the geometry code runs; calling the loader proves the
   whole path does — the fetch, the token re-check, the DOM assembly. The bug
   that prompted this harness lived in the loader. */

await checkAsync("refreshAll (state, ledger, and the level it selects)", async () => {
  await api.refreshAll();
  if ($("#levelTree").childElementCount === 0) throw new Error("no levels rendered");
  if ($("#ledger").childElementCount === 0) throw new Error("no ledger rendered");
});

await checkAsync("loadScan + renderPlateaus", async () => {
  await api.loadScan();
  if (marks("#scanChart") === 0) throw new Error("scan drew nothing");
  if ($("#plateaus").textContent.trim() === "") throw new Error("no plateaus rendered");
});

await checkAsync("applyThreshold (preview + cluster swatches)", async () => {
  await api.applyThreshold();
  if ($("#clusterList").childElementCount === 0) throw new Error("no swatches");
});

await checkAsync("loadCoverage", async () => {
  await api.loadCoverage();
  if ($("#coverage").textContent.trim() === "") throw new Error("rendered nothing");
});

await checkAsync("loadAxis", async () => {
  await api.loadAxis();
  if ($("#axis").textContent.trim() === "") throw new Error("rendered nothing");
});

await checkAsync("loadSupport", async () => {
  await api.loadSupport();
  if ($("#support").textContent.trim() === "") throw new Error("rendered nothing");
});

await checkAsync("loadHeatmap", async () => {
  await api.loadHeatmap();
  if ($("#heatmapChart").querySelectorAll("rect").length === 0)
    throw new Error("drew no cells");
  if ($("#heatmapChart").querySelectorAll("title").length === 0)
    throw new Error("cells carry no tooltips");
});

await checkAsync("loadProfile", async () => {
  await api.loadProfile();
  if (marks("#profileChart") === 0) throw new Error("drew nothing");
});

await checkAsync("loadFlow", async () => {
  await api.loadFlow();
  if (marks("#flowChart") === 0) throw new Error("drew nothing");
});

await checkAsync("loadMap", async () => {
  const before = calls.fillRect;
  await api.loadMap();
  if (calls.fillRect === before) throw new Error("painted no points");
});

await checkAsync("loadFeatureScatter", async () => {
  const before = calls.fillRect;
  await api.loadFeatureScatter();
  if (calls.fillRect === before) throw new Error("painted no points");
});

await checkAsync("loadBoundary", async () => {
  await api.loadBoundary();
  if ($("#boundary").textContent.trim() === "") throw new Error("rendered nothing");
});

await checkAsync("loadGradient (with the dimension profile chart)", async () => {
  await api.loadGradient();
  if ($("#gradient").textContent.trim() === "") throw new Error("rendered nothing");
  // the chart sits behind a `?.length` guard, so an empty fixture would skip it
  // silently and this check would pass without covering anything
  if ($("#gradient").querySelectorAll("circle").length === 0)
    throw new Error("no dimension-profile points drawn (empty profile fixture?)");
  const verdict = $("#gradient").textContent;
  if (!/scale|dimension/i.test(verdict)) throw new Error("no profile verdict text");
});

await checkAsync("loadSoftLabels payload is reachable", async () => {
  const res = await window.fetch(`/api/levels/${fixtures.level}/soft-labels`);
  JSON.parse(await res.text());
});

await checkAsync("watchJobs renders the job pill", async () => {
  api.watchJobs();
  await new Promise(resolve => window.setTimeout(resolve, 40));
  api.stopJobs();
  if ($("#jobPill").textContent.trim() === "") throw new Error("pill is empty");
});

await checkAsync("buttons enable once a level has a clustering", async () => {
  const gated = ["#btnRestrict", "#btnAxis", "#btnSupport", "#btnFlow", "#btnEmbed"];
  const level = fixtures.state.levels.find(l => l.has_clustering);
  if (!level) throw new Error("fixture has no clustered level to test with");

  api.S.level = level.name;
  await api.renderLevel();
  const stuck = gated.filter(sel => $(sel).disabled);
  if (stuck.length) throw new Error(`still disabled when clustered: ${stuck.join(", ")}`);

  // and they must go back off for a level that has not been swept
  const fresh = { ...level, name: "__fresh__", has_clustering: false, n_cells: null,
                  cut: { distance_threshold: null, min_cluster_size: 1, names: {}, colors: {} } };
  api.S.state.levels = [...fixtures.state.levels, fresh];
  api.S.level = "__fresh__";
  await api.renderLevel();
  const live = gated.filter(sel => !$(sel).disabled);
  if (live.length) throw new Error(`enabled with no clustering: ${live.join(", ")}`);
  api.S.state.levels = fixtures.state.levels;
  api.S.level = fixtures.level;
  await api.renderLevel();
});

// Variants the loaders cannot reach, driven directly.
check("drawProfile (raw counts variant)", () => {
  api.drawProfile(fixtures.profile, false);
  if (marks("#profileChart") === 0) throw new Error("drew nothing");
});

check("drawMap (isolating a cluster)", () => {
  const before = calls.fillRect;
  api.S.cluster = fixtures.scatter.cluster[0];
  api.setMapData(fixtures.scatter);
  api.drawMap();
  api.S.cluster = null;
  if (calls.fillRect === before) throw new Error("painted no points");
});

check("drawMap (by stability)", () => {
  api.setMapData(fixtures.stability);
  api.drawMap();
});

check("staleness marking", () => {
  api.S.preview = fixtures.preview;
  api.drawScan();
  $("#scanChart").dataset.token = api.viewToken();
  api.S.preview = { ...fixtures.preview, distance_threshold: 999 };
  api.markStale();
  if (!$("#scanChart").classList.contains("stale")) throw new Error("not marked stale");
  api.S.preview = fixtures.preview;
  api.markStale();
  if ($("#scanChart").classList.contains("stale")) throw new Error("stayed stale");
});

check("resetViews clears every derived panel", () => {
  api.drawProfile(fixtures.profile, true);
  api.resetViews();
  if (marks("#profileChart") !== 0) throw new Error("profile survived");
  if ($("#flowChart").innerHTML !== "") throw new Error("flow survived");
});

check("every fixture endpoint was actually exercised", () => {
  const unused = Object.keys(ROUTES).filter(
    k => fixtures[ROUTES[k]] !== undefined && !served.has(k));
  if (unused.length) throw new Error(`never fetched: ${unused.join(", ")}`);
});

if (failures.length) {
  process.stdout.write(`\n${failures.length} failure(s):\n`);
  for (const f of failures) process.stdout.write(`  - ${f}\n`);
  process.exit(1);
}
process.stdout.write("\nall renderers ran clean\n");
