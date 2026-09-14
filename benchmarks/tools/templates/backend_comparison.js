const rows = __PAYLOAD__;
const backends = ["cython_fallback", "rust_core"];
const colors = {cython_fallback: "#2f75b5", rust_core: "#d05a45"};
const linePalette = [
  "#2f75b5", "#d05a45", "#0f766e", "#9467bd", "#8c6d31", "#6b7280",
  "#1f9fb2", "#b44e9d", "#6f9e3f", "#d38b2f", "#4b5cc4", "#a14f3f"
];
const labels = {
  cython_fallback: "Cython fallback",
  rust_core: "Rust core"
};
const modeLabels = {
  bar: "Bar plot",
  line: "Line plot"
};
let xMode = "commit";
const metricUnits = {
  time: {label: "time", unit: "ms", factor: 1000},
  memory: {label: "peak memory", unit: "MiB", factor: 1 / (1024 * 1024)}
};

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function unique(values) {
  return Array.from(new Set(values)).sort((a, b) => {
    const ta = Number(String(a).replace("threads_", ""));
    const tb = Number(String(b).replace("threads_", ""));
    if (!Number.isNaN(ta) && !Number.isNaN(tb)) return ta - tb;
    return String(a).localeCompare(String(b));
  });
}

function optionText(kind, value) {
  if (kind === "commit") {
    const row = rows.find(item => item.commit === value);
    if (!row) return value;
    const tagPart = row.tag ? ` · ${row.tag}` : "";
    return `${value}${tagPart} (${row.date_label})`;
  }
  if (kind === "tag") {
    const row = rows.find(item => item.tag === value);
    return row ? `${value} — ${row.date_label}` : value;
  }
  return value;
}

function commitOrder(values) {
  return unique(values).sort((a, b) => {
    const ra = rows.find(item => item.commit === a);
    const rb = rows.find(item => item.commit === b);
    const da = ra ? ra.date : 0;
    const db = rb ? rb.date : 0;
    if (da !== db) return da - db;
    return String(a).localeCompare(String(b));
  });
}

function tagOrder(values) {
  return Array.from(new Set(values)).sort((a, b) => {
    const ra = rows.find(r => r.tag === a);
    const rb = rows.find(r => r.tag === b);
    const da = ra ? ra.date : 0;
    const db = rb ? rb.date : 0;
    if (da !== db) return da - db;
    return String(a).localeCompare(String(b));
  });
}

function checkedValues(id) {
  return Array.from(
    document.querySelectorAll(`#${id} input[type="checkbox"]:checked`)
  ).map(input => input.value);
}

function selectedOptions(id) {
  const el = document.getElementById(id);
  if (!el) return [];
  if (el.tagName === "SELECT") {
    return Array.from(el.selectedOptions).map(opt => opt.value);
  }
  return checkedValues(id);
}

function setSelectOptions(id, values, previousValues = [], textByValue = {}, defaultValues = null) {
  const el = document.getElementById(id);
  if (!el) return;
  const retained = values.filter(v => previousValues.includes(v));
  const defaults = defaultValues === null ? values : defaultValues;
  const selected = new Set(retained.length ? retained : defaults.filter(v => values.includes(v)));
  if (!selected.size && values.length) selected.add(values[0]);
  el.innerHTML = values.map(value => {
    const sel = selected.has(value) ? " selected" : "";
    const text = textByValue[value] || optionText(id, value);
    return `<option value="${escapeHtml(value)}"${sel}>${escapeHtml(text)}</option>`;
  }).join("");
}

function selectAllOptions(id) {
  const el = document.getElementById(id);
  if (!el) return;
  for (const opt of el.options) opt.selected = true;
  refreshOptions();
  renderAll();
}

function clearAllOptions(id) {
  const el = document.getElementById(id);
  if (!el) return;
  for (const opt of el.options) opt.selected = false;
  refreshOptions();
  renderAll();
}

function referenceModeValue() {
  return xMode;
}

function setReferenceMode(mode) {
  xMode = mode;
  document.getElementById("modeCommit").classList.toggle("active", mode === "commit");
  document.getElementById("modeTag").classList.toggle("active", mode === "tag");
  document.getElementById("commit").style.display = mode === "tag" ? "none" : "";
  document.getElementById("tag").style.display = mode === "tag" ? "" : "none";
  refreshOptions();
  renderAll();
}

function setCheckboxes(
  id,
  values,
  previousValues = [],
  textByValue = {},
  defaultValues = null
) {
  const container = document.getElementById(id);
  const retained = values.filter(value => previousValues.includes(value));
  const defaults = defaultValues === null ? values : defaultValues;
  const selected = new Set(retained.length ? retained : defaults.filter(value => values.includes(value)));
  if (!selected.size && values.length) selected.add(values[0]);
  container.innerHTML = values.map(value => {
    const checked = selected.has(value) ? " checked" : "";
    const text = textByValue[value] || optionText(id, value);
    return (
      `<label class="check-label">` +
      `<input type="checkbox" value="${escapeHtml(value)}"${checked}>` +
      `<span>${escapeHtml(text)}</span>` +
      `</label>`
    );
  }).join("");
}

function enforceSingleChoice(id, event) {
  if (!event.target.matches("input[type='checkbox']")) return;
  const inputs = Array.from(document.querySelectorAll(`#${id} input[type="checkbox"]`));
  if (event.target.checked) {
    for (const input of inputs) {
      if (input !== event.target) input.checked = false;
    }
    return;
  }
  if (!inputs.some(input => input.checked)) {
    event.target.checked = true;
  }
}

function enforceAtLeastOne(id, event) {
  if (!event.target.matches("input[type='checkbox']")) return;
  const inputs = Array.from(document.querySelectorAll(`#${id} input[type="checkbox"]`));
  if (!inputs.some(input => input.checked)) {
    event.target.checked = true;
  }
}

function metricValue() {
  return checkedValues("metric")[0] || "";
}

function familyValue() {
  return checkedValues("family")[0] || "";
}

function viewModeValue() {
  return checkedValues("viewMode")[0] || "line";
}

function selectedBenchmark() {
  const candidates = unique(rows
    .filter(row => row.metric === metricValue() && row.family === familyValue())
    .map(row => row.benchmark));
  return candidates[0] || "";
}

function benchmarkRows() {
  const benchmark = selectedBenchmark();
  return rows.filter(row =>
    row.metric === metricValue() &&
    row.family === familyValue() &&
    row.benchmark === benchmark
  );
}

function filteredRows() {
  const metric = metricValue();
  const family = familyValue();
  const benchmark = selectedBenchmark();
  const cases = checkedValues("case");
  const threads = checkedValues("threads");
  const selectedBackends = checkedValues("backend");
  const isTagMode = referenceModeValue() === "tag";
  const xFilter = isTagMode
    ? (row => row.tag && selectedOptions("tag").includes(row.tag))
    : (row => selectedOptions("commit").includes(row.commit));
  return rows.filter(row =>
    (!metric || row.metric === metric) &&
    (!family || row.family === family) &&
    (!benchmark || row.benchmark === benchmark) &&
    cases.includes(row.case) &&
    threads.includes(row.threads) &&
    xFilter(row) &&
    selectedBackends.includes(row.backend)
  );
}

function lastValue(values) {
  return values.length ? values[values.length - 1] : "";
}

function refreshOptions() {
  const previousMain = {
    metrics: checkedValues("metric"),
    families: checkedValues("family"),
    viewModes: checkedValues("viewMode"),
    cases: checkedValues("case"),
    threads: checkedValues("threads"),
    commits: selectedOptions("commit"),
    tags: selectedOptions("tag"),
    backends: checkedValues("backend")
  };
  const metricOptions = unique(rows.map(row => row.metric));
  const defaultMetric = metricOptions.includes("time") ? "time" : metricOptions[0];
  setCheckboxes("metric", metricOptions, previousMain.metrics, {}, [defaultMetric]);
  const familyOptions = unique(rows
    .filter(row => row.metric === metricValue())
    .map(row => row.family));
  const defaultFamily = familyOptions.includes("krige") ? "krige" : familyOptions[0];
  setCheckboxes("family", familyOptions, previousMain.families, {}, [defaultFamily]);
  setCheckboxes("viewMode", ["line", "bar"], previousMain.viewModes, modeLabels, ["line"]);
  const isTagMode = referenceModeValue() === "tag";
  const benchRows = benchmarkRows();
  const caseOptions = unique(benchRows.map(row => row.case));
  const defaultCase =
    caseOptions.find(v => v.includes("extra_large")) ||
    caseOptions.find(v => v.includes("sampled_15000")) ||
    caseOptions.find(v => v.includes("srf_unstructured")) ||
    caseOptions[0];
  setCheckboxes("case", caseOptions, previousMain.cases, {}, [defaultCase]);
  const selectedCases = checkedValues("case");
  const threadOptions = unique(benchRows
    .filter(row => selectedCases.includes(row.case))
    .map(row => row.threads));
  const defaultThread = threadOptions.includes("threads_1") ? "threads_1" : threadOptions[0];
  setCheckboxes("threads", threadOptions, previousMain.threads, {}, [defaultThread]);
  const selectedThreads = checkedValues("threads");
  const baseRows = benchRows.filter(row =>
    selectedCases.includes(row.case) && selectedThreads.includes(row.threads)
  );
  if (isTagMode) {
    const tagOptions = tagOrder(baseRows.filter(row => row.tag).map(row => row.tag));
    setSelectOptions("tag", tagOptions, previousMain.tags, {}, tagOptions);
    const selectedTags = selectedOptions("tag");
    const backendOptions = backends.filter(backend => baseRows.some(row =>
      row.backend === backend && row.tag && selectedTags.includes(row.tag)
    ));
    setCheckboxes("backend", backendOptions, previousMain.backends, labels, backendOptions);
  } else {
    const commitOptions = commitOrder(baseRows.map(row => row.commit));
    setSelectOptions("commit", commitOptions, previousMain.commits, {}, [lastValue(commitOptions)]);
    const selectedCommits = selectedOptions("commit");
    const backendOptions = backends.filter(backend => baseRows.some(row =>
      row.backend === backend && selectedCommits.includes(row.commit)
    ));
    setCheckboxes("backend", backendOptions, previousMain.backends, labels, backendOptions);
  }
  const benchmark = selectedBenchmark();
  const mode = viewModeValue() === "line" ? "Line plot" : "Bar plot";
  document.getElementById("chart-title").textContent =
    benchmark ? `${mode} - ${benchmark}` : mode;
}

function formatValue(row) {
  const unit = metricUnits[row.metric];
  return row.value * unit.factor;
}

function formatNumber(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function axisMax(values) {
  const maxValue = Math.max(...values, 0);
  return maxValue > 0 ? maxValue * 1.16 : 1;
}

function orderedKeys(values) {
  const seen = new Set();
  const keys = [];
  for (const value of values) {
    if (!seen.has(value)) {
      seen.add(value);
      keys.push(value);
    }
  }
  return keys;
}

function barGroupKey(row) {
  const xKey = referenceModeValue() === "tag" ? row.tag : row.commit;
  return `${row.case}|${row.threads}|${xKey}`;
}

function barGroupLabel(key, showThread, showCommit) {
  const [caseName, threads, commit] = key.split("|");
  const parts = [caseName];
  if (showThread) parts.push(threads);
  if (showCommit) parts.push(commit);
  return parts.join(" · ");
}

function selectionSummary(values, noun) {
  return values.length === 1 ? values[0] : `${values.length} ${noun}`;
}

function seriesLabel(key) {
  const [backend, caseName, threads] = key.split("|");
  return `${labels[backend] || backend} · ${caseName} · ${threads}`;
}

function chartSubtitle(data) {
  const cases = checkedValues("case");
  const threads = checkedValues("threads");
  const isTagMode = referenceModeValue() === "tag";
  const xValues = isTagMode ? selectedOptions("tag") : selectedOptions("commit");
  const noun = isTagMode ? "tags" : "commits";
  const selectedBackends = checkedValues("backend").map(value => labels[value] || value);
  return [
    data[0].benchmark,
    selectionSummary(cases, "cases"),
    selectionSummary(threads, "thread groups"),
    selectionSummary(xValues, noun),
    selectionSummary(selectedBackends, "backends")
  ].join(" · ");
}

function plotWidthFor(container, minimum, perItemWidth, itemCount) {
  const available = Math.max(0, Math.floor(container.clientWidth - 18));
  return Math.max(minimum, available, perItemWidth * itemCount + 180);
}

function barChartHeight(width) {
  return Math.max(580, Math.min(720, Math.round(width * 0.34)));
}

function lineChartHeight(width) {
  return Math.max(540, Math.min(680, Math.round(width * 0.30)));
}

function renderBarChart(data) {
  const chart = document.getElementById("chart");
  const legend = document.getElementById("legend");
  const metric = data[0].metric;
  const unit = metricUnits[metric];
  const selectedThreads = checkedValues("threads");
  const selectedCommits = selectedOptions("commit");
  const selectedBackends = checkedValues("backend");
  const showThread = selectedThreads.length > 1;
  const showCommit = selectedCommits.length > 1;
  const groups = orderedKeys(data.map(barGroupKey));
  const values = data.map(formatValue);
  const maxValue = axisMax(values);
  const margin = {top: 60, right: 34, bottom: 124, left: 88};
  const backendCount = Math.max(1, selectedBackends.length);
  const barGap = 0;
  const barWidth = backendCount === 1 ? 48 : 40;
  const clusterWidth = backendCount * barWidth + (backendCount - 1) * barGap;
  const groupStep = clusterWidth + 34;
  const availableWidth = Math.max(960, Math.floor(chart.clientWidth - 18));
  const requiredWidth = margin.left + groups.length * groupStep + margin.right;
  const width = Math.max(availableWidth, requiredWidth);
  const height = barChartHeight(width);
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const usedPlotWidth = groups.length * groupStep;
  const plotLeft = margin.left + Math.max(0, (plotWidth - usedPlotWidth) / 2);
  const y = value => margin.top + plotHeight - (value / maxValue) * plotHeight;

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">`;
  svg += `<text x="${margin.left}" y="26" class="title">${escapeHtml(chartSubtitle(data))}</text>`;
  svg += `<text x="${margin.left}" y="45" class="axis-label">${unit.label} (${unit.unit}), lower is better</text>`;

  for (let i = 0; i <= 5; i++) {
    const value = (maxValue / 5) * i;
    const yy = y(value);
    svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${yy}" y2="${yy}" class="grid" />`;
    svg += `<text x="${margin.left - 10}" y="${yy + 4}" text-anchor="end" class="tick">${formatNumber(value)}</text>`;
  }

  groups.forEach((groupKey, groupIndex) => {
    const groupX = plotLeft + groupIndex * groupStep + groupStep / 2;
    const label = barGroupLabel(groupKey, showThread, showCommit);
    svg += `<text x="${groupX}" y="${height - 46}" text-anchor="middle" class="tick" transform="rotate(-25 ${groupX} ${height - 46})">${escapeHtml(label)}</text>`;
    selectedBackends.forEach((backend, backendIndex) => {
      const row = data.find(item => barGroupKey(item) === groupKey && item.backend === backend);
      if (!row) return;
      const value = formatValue(row);
      const x = groupX - clusterWidth / 2 + backendIndex * (barWidth + barGap) + barWidth / 2;
      const yy = y(value);
      const barHeight = margin.top + plotHeight - yy;
      svg += `<rect x="${x - barWidth / 2}" y="${yy}" width="${barWidth}" height="${barHeight}" rx="3" fill="${colors[backend]}" opacity="0.86"><title>${escapeHtml(label)} · ${labels[backend]}: ${formatNumber(value)} ${unit.unit}</title></rect>`;
      svg += `<text x="${x}" y="${yy - 8}" text-anchor="middle" class="bar-label">${formatNumber(value)}</text>`;
    });
  });

  svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${margin.top + plotHeight}" y2="${margin.top + plotHeight}" class="axis" />`;
  svg += `<line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${margin.top + plotHeight}" class="axis" />`;

  // % change annotations when exactly 2 commits/tags are compared
  const isTagMode = referenceModeValue() === "tag";
  const xVals = isTagMode
    ? tagOrder(selectedOptions("tag"))
    : commitOrder(selectedOptions("commit"));
  if (xVals.length === 2) {
    const baseVals = new Map();
    groups.forEach((groupKey, groupIndex) => {
      const parts = groupKey.split("|");
      const xKey = parts[2];
      const groupX = plotLeft + groupIndex * groupStep + groupStep / 2;
      selectedBackends.forEach((backend, backendIndex) => {
        const row = data.find(item => barGroupKey(item) === groupKey && item.backend === backend);
        if (!row) return;
        const value = formatValue(row);
        const barX = groupX - clusterWidth / 2 + backendIndex * (barWidth + barGap) + barWidth / 2;
        const key = `${parts[0]}|${parts[1]}|${backend}`;
        if (xKey === xVals[0]) {
          baseVals.set(key, value);
        } else {
          const base = baseVals.get(key);
          if (!base) return;
          const pct = ((value - base) / base) * 100;
          const sign = pct >= 0 ? "+" : "";
          const fill = Math.abs(pct) <= 5 ? "var(--muted)" : pct > 0 ? "var(--danger)" : "var(--success)";
          const annotY = Math.max(margin.top + 4, y(value) - 20);
          svg += `<text x="${barX}" y="${annotY}" text-anchor="middle" font-size="10" fill="${fill}" font-weight="700">${sign}${pct.toFixed(1)}%</text>`;
        }
      });
    });
  }

  svg += `</svg>`;
  chart.innerHTML = svg;
  legend.innerHTML = selectedBackends.map(backend =>
    `<span class="legend-chip"><span class="legend-swatch" style="background:${colors[backend]}"></span>${escapeHtml(labels[backend] || backend)}</span>`
  ).join("");
}

function renderLineChart(data) {
  const chart = document.getElementById("chart");
  const legend = document.getElementById("legend");
  const metric = data[0].metric;
  const unit = metricUnits[metric];
  const isTagMode = referenceModeValue() === "tag";
  const rowXKey = row => isTagMode ? row.tag : row.commit;
  const xKeys = isTagMode
    ? tagOrder(data.map(rowXKey))
    : commitOrder(data.map(row => row.commit));
  const values = data.map(formatValue);
  const maxValue = axisMax(values);
  const width = plotWidthFor(chart, 960, 150, xKeys.length);
  const height = lineChartHeight(width);
  const margin = {top: 60, right: 42, bottom: 108, left: 88};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = key => {
    if (xKeys.length === 1) return margin.left + plotWidth / 2;
    return margin.left + xKeys.indexOf(key) * (plotWidth / (xKeys.length - 1));
  };
  const y = value => margin.top + plotHeight - (value / maxValue) * plotHeight;
  const groups = new Map();
  for (const row of data) {
    const key = `${row.backend}|${row.case}|${row.threads}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">`;
  svg += `<text x="${margin.left}" y="26" class="title">${escapeHtml(chartSubtitle(data))}</text>`;
  svg += `<text x="${margin.left}" y="45" class="axis-label">${unit.label} (${unit.unit}), lower is better</text>`;
  for (let i = 0; i <= 5; i++) {
    const value = (maxValue / 5) * i;
    const yy = y(value);
    svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${yy}" y2="${yy}" class="grid" />`;
    svg += `<text x="${margin.left - 10}" y="${yy + 4}" text-anchor="end" class="tick">${formatNumber(value)}</text>`;
  }
  xKeys.forEach(xKey => {
    const xx = x(xKey);
    const row = rows.find(item => rowXKey(item) === xKey);
    svg += `<text x="${xx}" y="${height - 56}" text-anchor="middle" class="tick" transform="rotate(-25 ${xx} ${height - 56})">${escapeHtml(xKey)}</text>`;
    if (row && row.date_label !== "-") {
      svg += `<text x="${xx}" y="${height - 30}" text-anchor="middle" class="tick">${escapeHtml(row.date_label)}</text>`;
    }
  });

  Array.from(groups.entries()).forEach(([key, group], index) => {
    const color = linePalette[index % linePalette.length];
    const byXKey = new Map(group.map(row => [rowXKey(row), row]));
    const points = xKeys
      .filter(xKey => byXKey.has(xKey))
      .map(xKey => {
        const row = byXKey.get(xKey);
        return {x: x(xKey), y: y(formatValue(row)), row, xKey};
      });
    if (!points.length) return;
    svg += `<polyline class="line-path" stroke="${color}" points="${points.map(point => `${point.x},${point.y}`).join(" ")}"><title>${escapeHtml(seriesLabel(key))}</title></polyline>`;
    for (const point of points) {
      const value = formatValue(point.row);
      svg += `<circle class="point" cx="${point.x}" cy="${point.y}" r="3.5" stroke="${color}"><title>${escapeHtml(seriesLabel(key))} · ${escapeHtml(point.xKey)}: ${formatNumber(value)} ${unit.unit}</title></circle>`;
    }
  });

  svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${margin.top + plotHeight}" y2="${margin.top + plotHeight}" class="axis" />`;
  svg += `<line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${margin.top + plotHeight}" class="axis" />`;
  svg += `</svg>`;
  chart.innerHTML = svg;
  legend.innerHTML = Array.from(groups.keys()).map((key, index) =>
    `<span class="legend-chip"><span class="legend-swatch" style="background:${linePalette[index % linePalette.length]}"></span>${escapeHtml(seriesLabel(key))}</span>`
  ).join("");
}

function renderChart() {
  const data = filteredRows();
  const chart = document.getElementById("chart");
  const legend = document.getElementById("legend");
  if (!data.length) {
    const emptyMsg = referenceModeValue() === "tag"
      ? "No benchmarked commits carry a tag in these results. " +
        "Tags appear once a tagged commit has been through the publish workflow."
      : "No matching ASV rows for this selection.";
    chart.innerHTML = `<p style="color:var(--muted);padding:12px 4px">${emptyMsg}</p>`;
    legend.innerHTML = "";
    return;
  }
  if (viewModeValue() === "line") {
    renderLineChart(data);
  } else {
    renderBarChart(data);
  }
}

function renderAll() {
  renderChart();
}

for (const id of ["metric", "family", "viewMode"]) {
  document.getElementById(id).addEventListener("change", event => {
    enforceSingleChoice(id, event);
    refreshOptions();
    renderAll();
  });
}
for (const id of ["case", "threads", "backend"]) {
  document.getElementById(id).addEventListener("change", event => {
    enforceAtLeastOne(id, event);
    refreshOptions();
    renderAll();
  });
}
for (const id of ["commit", "tag"]) {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => { refreshOptions(); renderAll(); });
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderAll, 120);
});
refreshOptions();
renderAll();
