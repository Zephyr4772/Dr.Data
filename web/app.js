/* DataDoctor front-end: intake → use case → examination → diagnosis report */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const E = Charts.esc, N = Charts.num;
const S = { ds: null, preset: null, uc: null, job: null, r: null, stages: [] };

// ============================================================ helpers
const pct = (v, d = 0) => v == null || !isFinite(v) ? '–' : (100 * v).toFixed(d) + '%';
const LOWER_BETTER = new Set(['mae', 'rmse', 'mape']);
const fm = (m, v) => v == null || !isFinite(v) ? '–' : m === 'mape' ? pct(v, 1) : N(v, 3);
const secs = s => s == null ? '–' : s < .001 ? '< 1 ms' : s < 1 ? `${(s * 1000).toFixed(0)} ms` : s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(0)} min` : `${(s / 3600).toFixed(1)} h`;
const tone = s => s >= 80 ? 'good' : s >= 55 ? 'warn' : 'bad';
const DIM_ORDER = ['volume', 'quality', 'labels', 'leakage', 'signal', 'validity', 'fairness'];
const SEVS = ['critical', 'major', 'minor', 'info'];
function toast(msg, ms = 3200) { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, ms); }
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch { } throw new Error(m); }
  return res.json();
}
function go(view) {
  $$('.view').forEach(v => v.hidden = v.id !== 'view-' + view);
  const order = ['intake', 'usecase', 'running', 'report'], i = order.indexOf(view);
  $$('#steps span').forEach((s, j) => {
    s.className = j === i ? 'on' : j < i ? 'done' : '';
    const can = (s.dataset.step === 'intake') || (s.dataset.step === 'usecase' && S.ds) || (s.dataset.step === 'report' && S.r);
    if (can && j !== i && s.dataset.step !== 'running') s.classList.add('clickable');
  });
  window.scrollTo({ top: 0 });
}
$$('#steps span').forEach(s => s.addEventListener('click', () => { if (s.classList.contains('clickable')) go(s.dataset.step); }));
$('#home').addEventListener('click', e => { e.preventDefault(); go('intake'); });

// theme
(() => {
  let saved = null; try { saved = localStorage.getItem('dd-theme'); } catch { }
  if (saved) document.documentElement.dataset.theme = saved;
  $('#themeBtn').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const nxt = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = nxt;
    try { localStorage.setItem('dd-theme', nxt); } catch { }
  });
})();

// ============================================================ 1. intake
async function loadDemos() {
  try {
    const demos = await api('/api/demos');
    $('#demoList').innerHTML = demos.map(d => `<button class="demo" data-id="${d.id}"><b>${E(d.name)}</b><span>${E(d.blurb)}</span><span class="tag">${d.task}</span></button>`).join('');
    $$('.demo').forEach(b => b.addEventListener('click', async () => {
      b.style.opacity = .5;
      try { const ds = await api('/api/demo/' + b.dataset.id, { method: 'POST' }); openDataset(ds); }
      catch (e) { toast(e.message); } finally { b.style.opacity = 1; }
    }));
  } catch (e) { $('#demoList').innerHTML = `<p class="muted">Server not reachable: ${E(e.message)}</p>`; }
}
const drop = $('#drop');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
$('#file').addEventListener('change', e => { if (e.target.files[0]) upload(e.target.files[0]); });
async function upload(f) {
  $('#dropMsg').textContent = '';
  $('#drop strong').textContent = `Reading ${f.name}…`;
  const fd = new FormData(); fd.append('file', f);
  try { openDataset(await api('/api/dataset', { method: 'POST', body: fd })); }
  catch (e) { $('#dropMsg').textContent = e.message; }
  finally { $('#drop strong').textContent = 'Drop a CSV / Excel file here'; $('#file').value = ''; }
}

// ============================================================ 2. use case
function openDataset(ds) {
  S.ds = ds; S.preset = ds.preset || null;
  const sug = ds.suggest, pre = ds.preset || {};
  S.uc = {
    target: pre.target || (sug.targets[0] || {}).column || ds.profile[ds.profile.length - 1].name,
    task: pre.task || 'auto', title: pre.title || '', goal: pre.goal || '', positive_class: pre.positive_class || null,
    interpretability: pre.interpretability || 'nice', deployment: pre.deployment || 'server',
    time_column: pre.time_column !== undefined ? pre.time_column : (sug.time_columns[0] || null),
    entity_column: pre.entity_column !== undefined ? pre.entity_column : (sug.entity_columns[0] || null),
    unavailable: pre.unavailable || [], group_columns: pre.group_columns || sug.group_columns.slice(0, 2), ignore: pre.ignore || [],
  };
  renderDsCard(); renderForm(); go('usecase');
}
const kindOf = name => (S.ds.profile.find(p => p.name === name) || {}).kind;
function renderDsCard() {
  const ds = S.ds;
  const kinds = {};
  ds.profile.forEach(p => kinds[p.kind] = (kinds[p.kind] || 0) + 1);
  $('#dsCard').innerHTML = `<p class="eyebrow">Dataset</p><h3>${E(ds.name)}</h3>
    <div class="ds-stats"><div><b>${ds.rows.toLocaleString()}</b><span>rows</span></div><div><b>${ds.cols}</b><span>columns</span></div>
    <div><b>${ds.notes.encoding}</b><span>encoding</span></div></div>
    <p class="small muted">${Object.entries(kinds).map(([k, v]) => `<span class="kind ${k}">${v} ${k}</span>`).join(' ')}</p>
    <ul class="collist">${ds.profile.map(p => `<li><span title="${E(p.name)}">${E(p.name)}</span><span><span class="kind ${p.kind}">${p.kind}</span> ${p.missing > 0 ? `<span class="small muted">${pct(p.missing)} ∅</span>` : ''}</span></li>`).join('')}</ul>`;
}
function seg(name, opts, val, cls = '') {
  return `<div class="seg ${cls}" data-name="${name}">${opts.map(([v, l, s]) => `<button type="button" data-v="${v}" class="${String(val) === String(v) ? 'on' : ''}">${l}${s ? `<small>${s}</small>` : ''}</button>`).join('')}</div>`;
}
function chipsFor(name, selected, filter = () => true, sugg = []) {
  return `<div class="chips" data-multi="${name}">${S.ds.profile.filter(p => p.name !== S.uc.target && filter(p)).map(p =>
    `<button type="button" class="chip ${selected.includes(p.name) ? 'on' : ''} ${sugg.includes(p.name) ? 'sug' : ''}" data-v="${E(p.name)}">${E(p.name)}</button>`).join('')}</div>`;
}
function renderForm() {
  const uc = S.uc, ds = S.ds, sug = ds.suggest;
  const tprof = ds.profile.find(p => p.name === uc.target) || {};
  const autoTask = (tprof.kind === 'numeric' && tprof.unique > 10) ? 'regression' : 'classification';
  const task = uc.task === 'auto' ? autoTask : uc.task;
  const goals = ds.goals[task];
  if (!goals[uc.goal]) uc.goal = Object.keys(goals)[0];
  const classes = (tprof.top || []).map(t => t[0]);
  const binary = task === 'classification' && tprof.unique === 2;
  const opt = (arr, val, none = true) => (none ? `<option value="">— none —</option>` : '') + arr.map(c => `<option ${c === val ? 'selected' : ''}>${E(c)}</option>`).join('');
  const cols = ds.profile.map(p => p.name).filter(c => c !== uc.target);
  $('#ucForm').innerHTML = `
    <p class="eyebrow">Step 2 · Describe the use case</p>
    <h2>What should a model do with this data?</h2>
    <p class="muted">The audit is judged against this. The same table can be fit for one use case and useless for another.</p>
    <div class="field"><label for="f-title">Use case<span class="help">One line, in business terms.</span></label>
      <input type="text" id="f-title" value="${E(uc.title)}" placeholder="e.g. Flag orders likely to be returned before shipping"></div>
    <div class="field"><label for="f-target">Target column<span class="help">The value you want to predict.</span></label>
      <div><select id="f-target">${opt(ds.profile.map(p => p.name), uc.target, false)}</select>
      <div class="hint-row">Suggested: ${sug.targets.slice(0, 6).map(t => `<button type="button" class="chip sug" data-target="${E(t.column)}" title="${E(t.why)}">${E(t.column)}</button>`).join(' ')}</div></div></div>
    <div class="field"><span class="flabel">Problem type<span class="help">Auto-detected from the target: <b>${autoTask}</b>.</span></span>
      ${seg('task', [['auto', 'Auto'], ['classification', 'Classification', 'predict a category'], ['regression', 'Regression', 'predict a number']], uc.task)}</div>
    <div class="field"><span class="flabel">What matters most?<span class="help">Decides the primary metric, the model choice and the decision threshold.</span></span>
      ${seg('goal', Object.entries(goals).map(([k, v]) => [k, v.split('(')[0].trim(), v.includes('(') ? v.split('(')[1].replace(')', '') : '']), uc.goal, 'col')}</div>
    ${binary ? `<div class="field"><label for="f-pos">Class of interest<span class="help">The "positive" outcome you want to catch.</span></label>
      <select id="f-pos">${opt(classes, uc.positive_class || '', true).replace('— none —', 'auto (rarer class)')}</select></div>` : ''}
    <div class="field"><label for="f-time">Time column<span class="help">If rows happen over time, the model will predict the future from the past. Enables forward-in-time validation and drift checks.</span></label>
      <select id="f-time">${opt(cols.filter(c => ['datetime', 'numeric'].includes(kindOf(c))), uc.time_column)}</select></div>
    <div class="field"><label for="f-ent">Entity column<span class="help">If several rows belong to the same customer / patient / device. Tests whether the model just memorises entities.</span></label>
      <select id="f-ent">${opt(cols.filter(c => ['identifier', 'categorical'].includes(kindOf(c))), uc.entity_column)}</select></div>
    <div class="field"><span class="flabel">Not known at prediction time<span class="help">Columns filled in after the moment you'd make the prediction (e.g. delivery days, rating, refund). These are the #1 source of leakage.</span></span>
      ${chipsFor('unavailable', uc.unavailable, p => !['identifier', 'constant'].includes(p.kind))}</div>
    <div class="field"><span class="flabel">Check fairness across<span class="help">Groups where performance must be similar (region, gender, segment…). Dashed = suggested.</span></span>
      ${chipsFor('group_columns', uc.group_columns, p => ['categorical', 'boolean'].includes(p.kind) && p.unique <= 30, sug.group_columns)}</div>
    <div class="field"><span class="flabel">Ignore entirely<span class="help">Identifiers are excluded automatically.</span></span>
      ${chipsFor('ignore', uc.ignore, p => p.kind !== 'identifier')}</div>
    <div class="field"><span class="flabel">Interpretability</span>
      ${seg('interpretability', [['not_needed', 'Not needed', 'accuracy first'], ['nice', 'Nice to have', 'prefer simple if tied'], ['required', 'Required', 'humans must read the model']], uc.interpretability)}</div>
    <div class="field"><span class="flabel">Deployment</span>
      ${seg('deployment', [['server', 'Server / API'], ['batch', 'Batch scoring', 'latency irrelevant'], ['edge', 'Edge / device', 'small & fast model']], uc.deployment)}</div>
    <div class="form-actions"><button type="button" class="secondary" id="backBtn">← Different dataset</button>
      <button type="submit" class="primary">Run the examination →</button></div>`;
  const f = $('#ucForm');
  $('#f-title').addEventListener('input', e => uc.title = e.target.value);
  $('#f-target').addEventListener('change', e => setTarget(e.target.value));
  $$('[data-target]', f).forEach(b => b.addEventListener('click', () => setTarget(b.dataset.target)));
  $$('.seg', f).forEach(s => $$('button', s).forEach(b => b.addEventListener('click', () => { uc[s.dataset.name] = b.dataset.v; if (s.dataset.name === 'task') uc.goal = ''; renderForm(); })));
  $$('[data-multi]', f).forEach(box => $$('.chip', box).forEach(b => b.addEventListener('click', () => {
    const arr = uc[box.dataset.multi], v = b.dataset.v, i = arr.indexOf(v);
    i >= 0 ? arr.splice(i, 1) : arr.push(v); b.classList.toggle('on');
  })));
  if ($('#f-pos')) $('#f-pos').addEventListener('change', e => uc.positive_class = e.target.value || null);
  $('#f-time').addEventListener('change', e => uc.time_column = e.target.value || null);
  $('#f-ent').addEventListener('change', e => uc.entity_column = e.target.value || null);
  $('#backBtn').addEventListener('click', () => go('intake'));
  f.onsubmit = e => { e.preventDefault(); runAnalysis(); };
}
function setTarget(t) {
  const uc = S.uc; uc.target = t; uc.task = 'auto'; uc.goal = ''; uc.positive_class = null;
  ['unavailable', 'group_columns', 'ignore'].forEach(k => uc[k] = uc[k].filter(c => c !== t));
  if (uc.time_column === t) uc.time_column = null;
  if (uc.entity_column === t) uc.entity_column = null;
  renderForm();
}

// ============================================================ 3. running
async function runAnalysis() {
  const uc = { ...S.uc };
  if (uc.task === 'auto') uc.task = 'auto';
  let j;
  try { j = await api('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset_id: S.ds.dataset_id, usecase: uc }) }); }
  catch (e) { toast(e.message); return; }
  S.stages = j.stages; S.job = j.job_id;
  $('#runTitle').textContent = `Examining ${S.ds.name} for “${uc.title || 'predict ' + uc.target}”`;
  go('running');
  const t0 = Date.now();
  const tick = async () => {
    let s;
    try { s = await api('/api/job/' + S.job); } catch (e) { toast(e.message); return; }
    $('#stageList').innerHTML = S.stages.map(([k, label]) => {
      const st = s.done.includes(k) || s.status === 'done' ? 'done' : s.stage === k ? 'on' : '';
      return `<li class="${st}"><span class="st"></span><span>${E(label)}${st === 'on' && s.sub ? `<small>${E(s.sub)}</small>` : ''}</span></li>`;
    }).join('');
    $('#runClock').textContent = `${((Date.now() - t0) / 1000).toFixed(0)} s elapsed`;
    if (s.status === 'done') { S.r = s.result; history.replaceState(null, '', '#job=' + S.job); renderReport(); go('report'); }
    else if (s.status === 'error') { toast('Analysis failed: ' + s.error, 8000); go('usecase'); }
    else setTimeout(tick, 600);
  };
  tick();
}

// ============================================================ 4. report
function section(id, num, title, intro, body) {
  return `<section id="sec-${id}"><div class="sec-head"><h2>${title}</h2><span class="num">${num}</span></div>${intro ? `<p class="sec-intro">${intro}</p>` : ''}${body}</section>`;
}
function explain(html) { return `<div class="explain">${html}</div>`; }
function renderReport() {
  const r = S.r;
  const parts = [
    ['dx', 'Diagnosis', secDiagnosis],
    ['findings', 'Findings', secFindings],
    ['data', 'Data profile', secData],
    ['target', 'Target', secTarget],
    ['signal', 'Signal & leakage', secSignal],
    ['validity', 'Validation realism', secValidity],
    ['models', 'Model bake-off', secModels],
    ['subgroups', 'Coverage & subgroups', secSubgroups],
    ['training', 'Training forecast', secTraining],
    ['labels', 'Label audit', secLabels],
    ['plan', 'Action plan', secPlan],
    ['method', 'Methodology', secMethod],
  ];
  let html = '';
  parts.forEach(([id, title, fn], i) => { try { html += fn(r, String(i + 1).padStart(2, '0')); } catch (e) { console.error(id, e); html += section(id, '', title, `<span class="muted">Could not render: ${E(e.message)}</span>`, ''); } });
  $('#reportBody').innerHTML = html;
  const counts = r.findings.filter(f => f.severity !== 'info').length;
  $('#toc').innerHTML = parts.map(([id, t]) => `<a href="#sec-${id}" data-id="${id}"><span>${t}</span>${id === 'findings' ? `<span class="n">${counts}</span>` : ''}</a>`).join('') +
    `<div class="toc-actions"><button class="secondary" id="pdfBtn">Export PDF</button><button class="secondary" id="jsonBtn">Download JSON</button><button class="secondary" id="editBtn">Edit use case</button></div>`;
  $('#pdfBtn').onclick = () => window.print();
  $('#jsonBtn').onclick = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(r, null, 1)], { type: 'application/json' })); a.download = 'datadoctor_report.json'; a.click(); };
  $('#editBtn').onclick = () => go('usecase');
  wireFindings(); wireCode(); wireToc();
}
window.addEventListener('beforeprint', () => $$('details.finding').forEach(d => d.open = true));

function wireToc() {
  const links = $$('#toc a');
  const obs = new IntersectionObserver(ents => ents.forEach(en => { if (en.isIntersecting) links.forEach(a => a.classList.toggle('on', a.dataset.id === en.target.id.slice(4))); }), { rootMargin: '-30% 0px -60% 0px' });
  $$('#reportBody > section').forEach(s => obs.observe(s));
}

// ------------------------------------------------------------ 01 diagnosis
function secDiagnosis(r, n) {
  const F = r.fitness, dims = F.dimensions;
  const axes = DIM_ORDER.map(k => ({ label: dims[k].label, value: dims[k].score / 100, tone: tone(dims[k].score) }));
  const rec = r.choice.recommended, bm = r.bakeoff.models[rec], m = r.metric;
  const sev = s => r.findings.filter(f => f.severity === s).length;
  const tiles = `<div class="tiles">
    <div class="tile"><div class="k">Rows · features</div><div class="v">${r.load.rows.toLocaleString()}</div><div class="d">${r.features.length} usable of ${r.load.cols - 1} columns</div></div>
    <div class="tile"><div class="k">Signal vs chance</div><div class="v">${N(r.signal.real, 2)}</div><div class="d">${r.signal.metric.replace('_', ' ')} · shuffled ≈ ${N(r.signal.null.reduce((a, b) => a + b, 0) / r.signal.null.length, 2)}</div></div>
    <div class="tile"><div class="k">Expected ${E(r.metric_label.split(' (')[0])}</div><div class="v">${fm(m, bm.mean[m])}</div><div class="d">± ${fm(m, bm.std[m])} · baseline ${fm(m, r.bakeoff.baseline[m])}</div></div>
    <div class="tile"><div class="k">Findings</div><div class="v"><span style="color:var(--crit)">${sev('critical')}</span> · <span style="color:var(--major)">${sev('major')}</span> · <span style="color:var(--minor)">${sev('minor')}</span></div><div class="d">critical · major · minor</div></div></div>`;
  const dimRows = DIM_ORDER.map(k => { const d = dims[k]; return `<div class="dim"><span>${d.label}</span><span class="meter"><i class="${tone(d.score)}-bg" style="width:${Math.max(2, d.score)}%"></i></span><span class="mono">${d.score}</span><span class="q">${d.question}</span></div>`; }).join('');
  const auto = [...r.repairs, ...Object.entries(r.excluded).map(([c, w]) => `Excluded '${c}': ${w}.`)];
  return section('dx', n, 'Diagnosis', '', `
    <div class="dx">
      <div>${Charts.ring(F.overall, F.tone)}<p class="small muted" style="text-align:center">Fitness for <i>this</i> use case</p></div>
      <div><p class="eyebrow">${E(r.usecase.title || 'Predict ' + r.usecase.target)} · ${r.task}</p>
        <div class="dx-verdict ${F.tone}">${E(F.verdict)}</div>
        <div class="narr">${r.narrative.map(p => p.startsWith('The main') ? `<div>${p}</div>` : `<p>${p}</p>`).join('')}</div></div>
    </div>
    ${tiles}
    <h3 class="sub">Fitness by dimension</h3>
    <div class="dimgrid"><div class="radarbox">${Charts.radar({ axes, size: 380 })}</div>
      <div><div class="dims">${dimRows}</div>
        <p class="small muted" style="margin-top:12px">Each dimension is 100·exp(−penalty/90), where critical / major / minor findings add 45 / 20 / 7 penalty points (one critical → 61, one major → 80). Signal is also bounded by the measured skill over chance. ${F.caps.length ? `Overall capped because of: ${F.caps.join(', ')}.` : ''}</p></div></div>
    <div class="grid2" style="margin-top:8px">
      <div><h3 class="sub">Fit with your constraints</h3><div class="fitlines">${r.usecase_fit.map(l => `<div><span>${E(l.label)}</span><b>${E(l.value)}<em>${E(l.note)}</em></b></div>`).join('')}</div></div>
      <div><h3 class="sub">Done automatically before modelling</h3><ul class="repairs">${auto.map(a => `<li>${E(a)}</li>`).join('') || '<li>Nothing needed.</li>'}</ul></div>
    </div>`);
}

// ------------------------------------------------------------ 02 findings
function miniChart(c) {
  if (!c) return '';
  if (c.type === 'bars') return Charts.hbar({ labels: c.labels, values: c.values, fmt: v => N(v, 1), width: 460, unit: c.unit });
  if (c.type === 'hist') return Charts.hist({ counts: c.counts, edges: c.edges, width: 460, height: 150, xlabel: c.label });
  return '';
}
function secFindings(r, n) {
  const dims = r.dimensions_meta;
  const cards = r.findings.map(f => `<details class="finding ${f.severity}" data-sev="${f.severity}" data-dim="${f.dimension}" ${f.severity === 'critical' ? 'open' : ''}>
    <summary><span class="sev ${f.severity}">${f.severity}</span><span class="t">${E(f.title)}</span><span class="dimtag">${E(dims[f.dimension].label)}</span></summary>
    <div class="fbody">
      <div><h5>Evidence</h5><p>${E(f.evidence)}</p>${f.columns && f.columns.length ? `<p>${f.columns.slice(0, 10).map(c => `<span class="tag">${E(c)}</span>`).join(' ')}</p>` : ''}</div>
      <div><h5>Why it matters here</h5><p>${E(f.impact)}</p><h5>What to do</h5><p>${E(f.fix)}</p>${f.auto_handled ? `<p class="auto small">✓ ${E(f.auto_handled)}</p>` : ''}</div>
      ${f.chart ? `<div class="full fchart">${miniChart(f.chart)}</div>` : ''}
    </div></details>`).join('');
  const sevBtns = SEVS.map(s => `<button class="chip on" data-fsev="${s}">${s} · ${r.findings.filter(f => f.severity === s).length}</button>`).join('');
  const dimBtns = DIM_ORDER.filter(d => r.findings.some(f => f.dimension === d)).map(d => `<button class="chip" data-fdim="${d}">${dims[d].label}</button>`).join('');
  return section('findings', n, 'Findings', 'Every issue the examination found, most severe first. Each one says what was measured, why it matters for <i>your</i> use case, and what to do. Click a row to expand.',
    `<div class="filters">${sevBtns}<span style="width:12px"></span>${dimBtns}</div><div id="findingList">${cards || '<p>No findings.</p>'}</div>`);
}
function wireFindings() {
  const st = { sev: new Set(SEVS), dim: null };
  const apply = () => $$('#findingList .finding').forEach(el => el.hidden = !(st.sev.has(el.dataset.sev) && (!st.dim || st.dim === el.dataset.dim)));
  $$('[data-fsev]').forEach(b => b.onclick = () => { const s = b.dataset.fsev; st.sev.has(s) ? st.sev.delete(s) : st.sev.add(s); b.classList.toggle('on'); apply(); });
  $$('[data-fdim]').forEach(b => b.onclick = () => { st.dim = st.dim === b.dataset.fdim ? null : b.dataset.fdim; $$('[data-fdim]').forEach(x => x.classList.toggle('on', x.dataset.fdim === st.dim)); apply(); });
}

// ------------------------------------------------------------ 03 data profile
function secData(r, n) {
  const issues = {};
  r.findings.forEach(f => (f.columns || []).forEach(c => { if (f.severity !== 'info') issues[c] = (issues[c] || 0) + 1; }));
  const rows = r.profile.map(p => {
    const excl = r.excluded[p.name], isT = p.name === r.usecase.target;
    const dist = p.hist ? Charts.spark(p.hist.counts) : p.top ? Charts.spark(p.top.map(t => t[1]), 'cat') : '';
    const range = p.kind === 'numeric' && p.min != null ? `${N(p.min)} … ${N(p.max)}` : p.kind === 'datetime' ? `${p.min} … ${p.max}` : p.top ? E(p.top.slice(0, 3).map(t => t[0]).join(', ')) : '';
    const status = isT ? '<b>target</b>' : excl ? `<span class="muted">excluded: ${E(excl)}</span>` : issues[p.name] ? `<span style="color:var(--major)">${issues[p.name]} issue${issues[p.name] > 1 ? 's' : ''}</span>` : '<span style="color:var(--good)">ok</span>';
    return `<tr class="${excl ? 'excl' : ''}"><td><b>${E(p.name)}</b></td><td><span class="kind ${p.kind}">${p.kind}</span></td><td class="n">${pct(p.missing, 1)}</td><td class="n">${p.unique.toLocaleString()}</td><td>${dist}</td><td class="small">${range}</td><td class="small">${status}</td></tr>`;
  }).join('');
  const mx = r.matrix;
  return section('data', n, 'Data profile', `What each column actually is (inferred from the values, not the file's types), how complete it is and what happens to it. ${mx.rows_used < mx.rows_total ? `Model-based checks use a representative sample of ${mx.rows_used.toLocaleString()} of ${mx.rows_total.toLocaleString()} rows${r.split_kind === 'time' ? ' kept in time order' : r.task === 'classification' ? ' stratified by class' : ''}.` : ''} After encoding, the ${mx.raw_features} feature columns become ${mx.encoded_features} numeric inputs.`,
    `<div class="tbl-wrap"><table class="t"><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Distribution</th><th>Range / top values</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${r.derived && r.derived.length ? `<h3 class="sub">Columns that are formulas of other columns</h3><ul class="notes">${r.derived.map(d => `<li><code>${E(d.formula)}</code> — holds for ${pct(d.share, 1)} of rows${d.violations ? `, <b>${d.violations} rows contradict it</b>` : ''}</li>`).join('')}</ul>` : ''}`);
}

// ------------------------------------------------------------ 04 target
function secTarget(r, n) {
  const t = r.target;
  let chart, text;
  if (t.type === 'classes') {
    const total = t.counts.reduce((a, b) => a + b, 0), mx = Math.max(...t.counts);
    chart = Charts.vbar({ labels: t.labels, values: t.counts, cls: i => t.labels[i] === t.positive ? 'bar-alt' : 'bar', ylabel: 'rows' });
    text = `<p>${t.labels.length} classes. The largest is ${pct(mx / total, 1)} of rows, so <b>always predicting it scores ${pct(mx / total, 1)} accuracy</b> while being useless${t.positive ? ` for finding '${E(t.positive)}' (highlighted)` : ''}. That is why this report uses <b>${E(r.metric_label)}</b>, chosen from your goal (“${E(r.goal_label)}”).</p>`;
  } else {
    chart = Charts.hist({ counts: t.hist, edges: t.edges, xlabel: r.usecase.target, marks: [{ x: t.median, label: 'median' }] });
    text = `<p>Mean ${N(t.mean)}, median ${N(t.median)}, std ${N(t.std)}, range ${N(t.min)} … ${N(t.max)}. Skewness ${N(t.skew, 2)}${Math.abs(t.skew) > 2 ? ' — a long tail: a few large values will dominate squared-error losses' : ''}.${t.zeros > .05 ? ` ${pct(t.zeros)} of values are exactly zero.` : ''}</p>
      <p>Primary metric: <b>${E(r.metric_label)}</b>, from your goal “${E(r.goal_label)}”. ${r.metric === 'rmse' ? 'RMSE squares errors, so large misses are punished hardest.' : r.metric === 'mae' ? 'MAE weighs every unit of error equally and is robust to the tail.' : 'MAPE measures error relative to the true value.'}</p>`;
  }
  const tf = r.findings.filter(f => f.dimension === 'labels' && f.severity !== 'info');
  return section('target', n, 'Target', 'The thing you want to predict decides which metrics are honest and which models fit.',
    `<div class="grid2"><div class="chartbox">${chart}</div><div>${text}${tf.length ? `<p class="small muted">Related findings: ${tf.map(f => E(f.title)).join(' · ')}</p>` : ''}</div></div>`);
}

// ------------------------------------------------------------ 05 signal & leakage
function secSignal(r, n) {
  const s = r.signal, sf = r.single_feature, rel = r.relevance;
  const mlab = { roc_auc: 'ROC-AUC', balanced_accuracy: 'Balanced accuracy', r2: 'R²' }[s.metric];
  const nullMean = s.null.reduce((a, b) => a + b, 0) / s.null.length;
  const sfe = Object.entries(sf.values).slice(0, 14);
  const leakSet = new Set(r.leak_removed || []);
  const thr = r.task === 'classification' ? [.7, .9] : [.85, .95];
  const imp = Object.entries(rel.importance).slice(0, 14);
  const verdict = s.skill < .03 ? `<b>No signal.</b> The real score sits inside the cloud of chance scores. The features cannot predict the target, so no model will either.` :
    s.skill < .15 ? `<b>Weak signal.</b> Real labels beat chance, but not by much (${pct(s.skill)} of the way from chance to perfect).` :
      `<b>Clear signal.</b> The real score is ${N(s.z, 0)} standard deviations above chance; ${pct(s.skill)} of the way from chance to perfect.`;
  return section('signal', n, 'Signal & leakage', 'Two opposite failure modes: features that carry <i>no</i> information about the target, and features that carry <i>too much</i> because they leak the answer.',
    `<h3 class="sub">Is there any real signal? (label-shuffle test)</h3>
    ${Charts.nullDist({ nul: s.null, real: s.real, label: mlab })}
    ${explain(`We train the same gradient-boosting model ${s.null.length + 1} times: once on the real labels and ${s.null.length} times on randomly shuffled labels, which destroys any real relationship. ${verdict} Shuffled mean ${N(nullMean, 3)}, real ${N(s.real, 3)}, permutation p ≤ ${N(s.p, 2)}.`)}
    <div class="grid2">
      <div class="chartbox"><h3 class="sub">Leakage scan: each column on its own</h3>
        ${sfe.length ? Charts.hbar({ labels: sfe.map(e => e[0]), values: sfe.map(e => e[1]), max: 1, min: 0, fmt: v => N(v, 2), cls: (i, v) => leakSet.has(sfe[i][0]) ? 'bar-crit' : v >= thr[0] ? 'bar-warn' : 'bar', ref: thr[1], refLabel: 'leak threshold' }) : '<p class="muted">No columns.</p>'}
        <p class="cap">${E(sf.measure)} using a small tree on one column at a time. Red = removed as leakage. A single column that nearly solves the problem is almost always recorded after, or derived from, the outcome.</p></div>
      <div class="chartbox"><h3 class="sub">Which features the model actually uses</h3>
        ${imp.length ? Charts.hbar({ labels: imp.map(e => e[0]), values: imp.map(e => e[1]), err: imp.map(e => rel.sd[e[0]]), fmt: v => N(v, 3), cls: (i, v) => rel.useless.includes(imp[i][0]) ? 'bar-muted' : 'bar' }) : ''}
        <p class="cap">Permutation importance: how much held-out ${E(rel.metric.replace('_', ' '))} drops when a column is shuffled (bars ± 1 sd). Grey = no measurable contribution. Computed after leaky columns were removed.</p></div>
    </div>`);
}

// ------------------------------------------------------------ 06 validation realism
function secValidity(r, n) {
  const v = r.validity, uc = r.usecase;
  const mlab = { roc_auc: 'ROC-AUC', balanced_accuracy: 'Balanced accuracy', r2: 'R²' }[v.metric];
  const labels = ['Random shuffle'], vals = [v.random];
  if (v.time != null) { labels.push('Past → future'); vals.push(v.time); }
  if (v.group != null) { labels.push(`Unseen ${uc.entity_column}`); vals.push(v.group); }
  let body = `<div class="grid2"><div class="chartbox"><h3 class="sub">Same model, different ways of splitting</h3>
    ${Charts.hbar({ labels, values: vals, fmt: x => N(x, 3), min: Math.min(0, ...vals), cls: i => i === 0 ? 'bar-muted' : 'bar' })}
    <p class="cap">${mlab} of gradient boosting under each validation scheme.</p></div><div>`;
  if (v.time != null) body += explain(`<b>Why this matters:</b> a random split mixes future rows into training. In production you only ever have the past. The drop from ${N(v.random, 3)} to ${N(v.time, 3)} is the optimism you would have reported. All model comparisons below use the honest scheme (${r.split_kind === 'time' ? 'past → future' : r.split_kind}).`);
  if (v.group != null) body += explain(`Each ${E(uc.entity_column)} appears ~${N(v.rows_per_entity, 1)} times. Grouped CV keeps all rows of an entity on one side, so the score reflects <b>new</b> entities.`);
  if (v.time == null && v.group == null) body += explain('No time or entity column was declared, so only random cross-validation is possible. If rows are ordered in time or several rows share an entity, declare it in the use case to test this.');
  body += `</div></div>`;
  if (v.drift) {
    const d = v.drift;
    body += `<div class="grid2"><div class="chartbox"><h3 class="sub">Is the target stable over time?</h3>
      ${Charts.line({ series: [{ name: d.what, x: d.values.map((_, i) => i), y: d.values, cls: 's2', dots: true }], xcat: d.labels, ylabel: d.what, legend: false, height: 220 })}
      <p class="cap">${E(d.what)} in 8 equal-size time periods (start date shown). Chi-square / Kruskal p = ${N(d.p, 3)}.</p></div>`;
    body += `<div class="chartbox"><h3 class="sub">Do the features drift? (adversarial validation)</h3>`;
    if (v.adversarial_auc != null) {
      body += `<p>A classifier tried to tell <b>older</b> rows from <b>newer</b> rows using only the features. ROC-AUC <b>${N(v.adversarial_auc, 2)}</b> ${v.adversarial_auc < .6 ? '(close to 0.5: the two periods look alike, good)' : v.adversarial_auc < .75 ? '(noticeable shift)' : '(strong shift: the future looks different from the past)'}.</p>`;
      if (v.drift_features) { const df = Object.entries(v.drift_features); body += Charts.hbar({ labels: df.map(e => e[0]), values: df.map(e => e[1]), fmt: x => N(x, 3), width: 520, cls: () => 'bar-warn' }) + '<p class="cap">Features that most give away the time period: monitor these in production.</p>'; }
    }
    body += `</div></div>`;
  }
  return section('validity', n, 'Validation realism', 'A model is only as good as the test that measured it. These checks ask whether a standard random split would give you a number you can trust.', body);
}

// ------------------------------------------------------------ 07 models
function secModels(r, n) {
  const b = r.bakeoff, m = r.metric, ch = r.choice, low = LOWER_BETTER.has(m);
  const names = ch.ranking;
  const allV = names.map(k => b.models[k].mean[m]);
  const maxV = Math.max(...allV, b.baseline[m]);
  const sec = r.task === 'classification' ? (r.classes.length === 2 ? ['roc_auc', 'recall', 'precision'] : ['roc_auc', 'f1', 'accuracy']) : ['r2', 'rmse', 'mae'].filter(x => x !== m).slice(0, 2);
  const rows = names.map(k => {
    const x = b.models[k], v = x.mean[m];
    return `<tr class="${k === ch.recommended ? 'rec' : ''}"><td><b>${E(k)}</b>${k === ch.recommended ? '<span class="recbadge">recommended</span>' : ''}<div class="small muted">${E(x.about)}</div></td>
      <td class="n"><span class="mbar" style="width:${Math.max(2, 70 * v / maxV)}px"></span>${fm(m, v)} <span class="muted">± ${fm(m, x.std[m])}</span></td>
      <td class="n">${x.skill > 0 ? pct(x.skill) : '<span style="color:var(--crit)">none</span>'}</td>
      ${sec.map(s => `<td class="n">${fm(s, x.mean[s])}</td>`).join('')}
      <td class="n">${secs(x.fit_time)}</td><td class="n">${N(x.latency_ms, 3)} ms</td><td class="n">${x.size_kb >= 1024 ? N(x.size_kb / 1024, 1) + ' MB' : N(x.size_kb, 0) + ' KB'}</td>
      <td><span class="dots" title="interpretability">${'●'.repeat(x.interp)}${'○'.repeat(3 - x.interp)}</span></td></tr>`;
  }).join('');
  const splitTxt = { time: 'forward-in-time folds (TimeSeriesSplit)', group: `folds grouped by ${r.usecase.entity_column}`, random: r.task === 'classification' ? 'stratified folds' : 'shuffled folds' }[b.split];
  const recM = b.models[ch.recommended];
  const foldSeries = names.slice(0, 4).map((k, i) => ({ name: k, x: b.models[k].folds[m].map((_, j) => j + 1), y: b.models[k].folds[m], cls: 's' + (i + 1), dots: true }));
  let th = '';
  if (r.threshold) {
    const t = r.threshold;
    th = `<h3 class="sub">Choosing the decision threshold</h3><div class="grid2"><div class="chartbox">
      ${Charts.line({ series: [{ name: 'precision', x: t.thresholds, y: t.precision, cls: 's1' }, { name: 'recall', x: t.thresholds, y: t.recall, cls: 's2' }, { name: `F${t.beta === 1 ? 1 : t.beta === 2 ? 2 : '0.5'}`, x: t.thresholds, y: t.f, cls: 's3', dash: true }],
      vlines: [{ x: t.best, label: `best ${N(t.best, 2)}`, cls: 'good' }, { x: .5, label: '0.5' }], xlabel: 'probability threshold for “' + r.positive + '”', ylo: 0, yhi: 1, xfmt: v => N(v, 1), height: 240 })}</div>
      <div>${explain(`Models output a probability; you choose where to cut. The default 0.5 is arbitrary. At 0.5 the recommended model gets precision <b>${pct(t.default_row[1])}</b> and recall <b>${pct(t.default_row[2])}</b>. Because your goal is “${E(r.goal_label.toLowerCase())}”, the threshold that maximises F${t.beta === 2 ? '2' : t.beta === 1 ? '1' : '0.5'} is <b>${N(t.best, 2)}</b>: precision ${pct(t.best_row[1])}, recall ${pct(t.best_row[2])}, flagging ${t.best_row[4]} of ${t.n} rows.`)}
      <p class="small muted">Computed on out-of-fold predictions. In the real pipeline, tune it on the validation set and freeze it before touching the test set.</p></div></div>`;
  }
  return section('models', n, 'Model bake-off', `Six model families trained and scored the same way: ${b.folds} ${splitTxt}, primary metric <b>${E(r.metric_label)}</b> (${low ? 'lower' : 'higher'} is better). The no-skill baseline (best possible without looking at any feature) scores <b>${fm(m, b.baseline[m])}</b>.${b.balanced_weights ? ' Class weights were used to counter imbalance.' : ''}`,
    `<div class="tbl-wrap"><table class="t"><thead><tr><th>Model</th><th>${E(r.metric_label.split(' (')[0])}</th><th>Skill</th>${sec.map(s => `<th>${s.replace('_', ' ')}</th>`).join('')}<th>Fit time</th><th>Latency</th><th>Size</th><th>Readable</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="small muted" style="margin-top:8px">Skill = share of the gap between the no-skill baseline and a perfect score that the model closes. Fit time is per fold on ${recM.train_rows.toLocaleString()} rows on this machine.</p>
    <div class="grid2"><div><h3 class="sub">Why ${E(ch.recommended)}</h3><ul class="notes">${ch.reasons.map(x => `<li>${E(x)}</li>`).join('')}</ul></div>
    <div class="chartbox"><h3 class="sub">Score per fold (stability)</h3>${Charts.line({ series: foldSeries, xlabel: 'fold', ylabel: r.metric_label.split(' (')[0], height: 220, xfmt: v => N(v, 0), yfmt: v => fm(m, v) })}
    <p class="cap">${b.split === 'time' ? 'Fold 1 trains on the least history, fold 5 on the most. Rising lines = the model benefits from more history.' : 'Large swings between folds = the score is noisy; small differences between models are not meaningful.'}</p></div></div>
    ${th}`);
}

// ------------------------------------------------------------ 08 subgroups
function secSubgroups(r, n) {
  const sg = r.subgroups, m = sg.metric;
  const cols = Object.entries(sg.columns);
  if (!cols.length) return section('subgroups', n, 'Coverage & subgroups', 'No group columns were selected or found.', '');
  const reliable = r.signal.skill >= .1;
  const body = cols.map(([c, rows]) => {
    const ok = rows.filter(x => x.score != null);
    const lowB = LOWER_BETTER.has(m);
    return `<div class="chartbox"><h3 class="sub">${E(c)}</h3>${Charts.hbar({ labels: ok.map(x => `${x.level} (${x.n})`), values: ok.map(x => x.score), fmt: v => fm(m, v), ref: sg.overall, refLabel: 'overall', min: lowB ? 0 : Math.min(0, ...ok.map(x => x.score)), width: 560, cls: (i, v) => (lowB ? v > sg.overall * 1.3 : v < sg.overall - .08) && ok[i].n >= 50 ? 'bar-warn' : 'bar' })}
      <p class="cap">${E(m.replace('_', ' '))} per group (out-of-fold). Group share of data: ${rows.map(x => `${E(x.level)} ${pct(x.share)}`).join(' · ')}${rows[0].rate != null ? '. ' + (r.task === 'regression' ? 'Mean target: ' : `Rate of '${E(r.positive)}': `) + rows.map(x => `${E(x.level)} ${r.task === 'regression' ? N(x.rate) : pct(x.rate, 1)}`).join(' · ') : ''}.</p></div>`;
  }).join('');
  return section('subgroups', n, 'Coverage & subgroups', `An average score can hide groups where the model fails. ${sg.auto ? 'No groups were selected, so the first low-cardinality columns were used.' : 'Groups you selected:'} Groups with fewer than 15 rows are not scored.`,
    (reliable ? '' : explain('<b>Caution:</b> the model learned little or nothing overall, so differences between groups here are noise, not bias.')) + `<div class="grid2">${body}</div>`);
}

// ------------------------------------------------------------ 09 training forecast
function curveChart(c, title) {
  return Charts.line({ series: [{ name: 'train loss', x: c.x, y: c.train, cls: 's2' }, { name: 'validation loss', x: c.x, y: c.val, cls: 's1' }],
    vlines: [{ x: c.best_x, label: `best @ ${c.best_x}`, cls: 'good' }], xlabel: c.xlabel, ylabel: c.loss, height: 250, yfmt: v => N(v, 3) });
}
function guideCurves(kind) {
  const x = Array.from({ length: 40 }, (_, i) => i + 1);
  const f = {
    healthy: [x.map(i => .25 + .7 * Math.exp(-i / 7)), x.map(i => .32 + .65 * Math.exp(-i / 8))],
    overfit: [x.map(i => .08 + .85 * Math.exp(-i / 6)), x.map(i => .38 + .6 * Math.exp(-i / 5) + .006 * Math.max(0, i - 10) ** 1.2)],
    underfit: [x.map(i => .78 + .15 * Math.exp(-i / 4)), x.map(i => .82 + .14 * Math.exp(-i / 4))],
    lr: [x.map(i => .5 + .3 * Math.exp(-i / 6) + .12 * Math.sin(i * 1.7) * Math.exp(-i / 60)), x.map(i => .55 + .3 * Math.exp(-i / 6) + .18 * Math.sin(i * 1.3 + 1))],
    still: [x.map(i => .95 - .015 * i), x.map(i => .98 - .014 * i)],
    leak: [x.map(i => .02 + .9 * Math.exp(-i)), x.map(i => .03 + .9 * Math.exp(-i))],
  }[kind];
  return Charts.line({ series: [{ name: '', x, y: f[0], cls: 's2' }, { name: '', x, y: f[1], cls: 's1' }], legend: false, height: 120, width: 300, ylo: 0, yhi: 1.05, yfmt: () => '', xfmt: () => '' });
}
function secTraining(r, n) {
  const T = r.training, tm = T.time, rec = T.recommended, ds = r.data_size;
  const g = T.boosting, mlp = T.mlp;
  const state = (r.signal.skill < .03) ? 'underfit' : g.state === 'overfits' ? 'overfit' : g.state === 'still_improving' ? 'still' : 'healthy';
  const cards = [
    ['healthy', 'Healthy', 'Both curves fall and flatten with a small, stable gap.', 'Stop where validation flattens; you are done.'],
    ['overfit', 'Overfitting', 'Train keeps falling, validation turns back up (U-shape).', 'Early stopping at the minimum, more regularisation, fewer features, more data.'],
    ['underfit', 'Under-fitting / no signal', 'Both curves stay high and flat, close together.', 'More capacity or better features. If the signal test failed, only new features help.'],
    ['lr', 'Learning rate too high', 'Jagged, oscillating validation loss; may diverge.', 'Lower the learning rate (÷3 to ÷10) or increase batch size.'],
    ['still', 'Still improving', 'Both curves still falling at the end.', 'Train longer or raise the learning rate a little.'],
    ['leak', 'Too good to be true', 'Validation loss drops to ~0 almost instantly.', 'Suspect leakage or duplicates across the split. Re-audit before celebrating.'],
  ].map(([k, t, see, act]) => `<div class="gcard ${k === state ? 'match' : ''}">${k === state ? '<span class="yours">↳ closest to your data</span>' : ''}<h5>${t}</h5>${guideCurves(k)}<p><b>Looks like:</b> ${see}</p><p><b>Do:</b> ${act}</p></div>`).join('');
  let lc = '';
  if (ds && ds.n && ds.n.length) {
    const m = ds.metric, series = [{ name: 'train', x: ds.n, y: ds.train, cls: 's2', dots: true }, { name: 'validation', x: ds.n, y: ds.val, cls: 's1', dots: true }];
    if (ds.extrapolation) series.push({ name: 'projected', x: [ds.n[ds.n.length - 1], ...ds.extrapolation.n.slice(1)], y: [ds.val[ds.val.length - 1], ...ds.extrapolation.score.slice(1)], cls: 's1', dash: true });
    lc = `<div class="grid2"><div class="chartbox">${Charts.line({ series, xlabel: 'training rows', ylabel: r.metric_label.split(' (')[0], height: 240, yfmt: v => fm(m, v) })}</div>
      <div>${explain(`<b>Will more data help?</b> ${rec} was retrained on 10% → 100% of the training folds. ${ds.extrapolation ? `An inverse power-law fit projects ${E(r.metric_label.split(' (')[0])} ≈ <b>${fm(m, ds.extrapolation.score[1])}</b> at ${Math.round(ds.extrapolation.n[1]).toLocaleString()} rows and ${fm(m, ds.extrapolation.score[2])} at ${Math.round(ds.extrapolation.n[2]).toLocaleString()} (dashed).` : ''} A curve that is still climbing means data-limited; a flat one means feature-limited. The gap between the train and validation curves is the variance (overfitting) part of the error.`)}</div></div>`;
  }
  return section('training', n, 'Training forecast', `What training will look like before you spend GPU/CPU time on it: real loss curves from short runs on your data, how long a proper run will take, and how to read the curves you will see.`,
    `<div class="grid3">
      <div class="tile"><div class="k">${E(rec)} · one fit, all ${tm.full_rows.toLocaleString()} rows</div><div class="v">${secs(tm.single_fit_full)}</div><div class="d">measured at 3 sizes, scales as n^${N(tm.exponent, 2)}</div></div>
      <div class="tile"><div class="k">5-fold cross-validation</div><div class="v">${secs(tm.cv5_full)}</div><div class="d">${tm.cpu} CPU cores on this machine</div></div>
      <div class="tile"><div class="k">Random search, 30 trials × 5 folds</div><div class="v">${secs(tm.search30_full)}</div><div class="d">10× more data: ${secs(tm.single_fit_10x)} per fit</div></div></div>
    <div class="grid2">
      <div class="chartbox"><h3 class="sub">Gradient boosting: loss per tree</h3>${curveChart(g)}
        <ul class="notes">${g.notes.map(x => `<li>${E(x)}</li>`).join('')}<li>Recommended: up to ~${Math.round(g.best_x * 1.5)} trees at learning rate ${g.lr}, early-stopping patience ≈ ${Math.max(10, g.patience)} rounds.</li></ul></div>
      <div class="chartbox"><h3 class="sub">Neural network (MLP 64-32): loss per epoch</h3>${curveChart(mlp)}
        <ul class="notes">${mlp.notes.map(x => `<li>${E(x)}</li>`).join('')}<li>Batch ${mlp.batch} → ${mlp.steps_per_epoch} steps/epoch, ${N(mlp.sec_per_epoch * 1000, 0)} ms/epoch here. Recommended: ~${Math.round(mlp.best_x * 1.2)} epochs with patience ≈ ${Math.max(5, mlp.patience)}.</li></ul></div>
    </div>
    <h3 class="sub">Data size vs performance</h3>${lc}
    <h3 class="sub">Loss-curve field guide</h3>
    <p class="small muted">Orange = training loss, teal = validation loss. Match what you see during training to a pattern below.</p>
    <div class="guide">${cards}</div>`);
}

// ------------------------------------------------------------ 10 labels
function secLabels(r, n) {
  const L = r.labels;
  if (!L) return section('labels', n, 'Label audit', 'Not available for this run.', '');
  const rows = (L.examples || []).map(e => `<tr>${L.columns.map(c => `<td class="${typeof e[c] === 'number' ? 'n' : ''}">${typeof e[c] === 'number' ? (c === 'row' ? e[c] : N(e[c], 2)) : E(e[c] ?? '')}</td>`).join('')}</tr>`).join('');
  const intro = L.type === 'classification'
    ? `Every row was scored by a model that never saw it in training (out-of-fold). Rows where the model gives <b>under 20%</b> probability to the recorded label while being confident in another class are likely mislabeled (simplified confident learning). Estimated suspect rate: <b>${pct(L.suspect_rate, 1)}</b> (${L.n_suspect} rows).`
    : `Rows whose actual target is more than 6 robust standard deviations from an out-of-fold prediction. These are either data errors in the target or rare events the features cannot see. Rate: <b>${pct(L.suspect_rate, 1)}</b> (${L.n_suspect} rows).`;
  return section('labels', n, 'Label audit', intro,
    (L.reliable === false ? explain('<b>Caution:</b> the model found little signal, so its disagreements with the labels are not evidence of label errors. Ignore this table until the signal problem is solved.') : '') +
    (rows ? `<div class="tbl-wrap"><table class="t"><thead><tr>${L.columns.map(c => `<th>${E(c.replace('_', ' '))}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div><p class="small muted" style="margin-top:8px">Row = index in your file after dropping exact duplicates and rows with no target. Showing up to 15, most suspicious first.</p>` : '<p>No suspicious rows found.</p>'));
}

// ------------------------------------------------------------ 11 plan
function secPlan(r, n) {
  const fmap = Object.fromEntries(r.findings.map((f, i) => [f.id, f]));
  const steps = r.plan.map(s => `<li><h4>${E(s.title)}</h4>${Array.isArray(s.detail) ? `<ul>${s.detail.map(d => `<li>${E(d)}</li>`).join('')}</ul>` : `<p>${E(s.detail)}</p>`}
    ${s.refs && s.refs.length ? `<p class="small muted">Addresses: ${s.refs.slice(0, 4).map(id => fmap[id] ? E(fmap[id].title) : '').filter(Boolean).join(' · ')}</p>` : ''}</li>`).join('');
  return section('plan', n, 'Action plan', 'The route from this dataset to a model you can defend, in order. Generated from the findings above.',
    `<ol class="plan">${steps}</ol><h3 class="sub">Starter code</h3><p class="small muted">A scikit-learn pipeline that implements the plan: cleaning, honest validation, the recommended model and your metric.</p>
    <div class="codewrap"><button id="copyBtn">Copy</button><pre class="code"><code id="codeBlock">${E(r.code)}</code></pre></div>`);
}
function wireCode() { const b = $('#copyBtn'); if (b) b.onclick = () => { navigator.clipboard.writeText(S.r.code).then(() => toast('Copied')); }; }

// ------------------------------------------------------------ 12 method
function secMethod(r, n) {
  const items = [
    ['Column typing', 'Each column is typed from its values (numbers stored as text, day-first dates, times, IDs, free text, PII by name/pattern), not from the file\'s dtypes.'],
    ['Hygiene', 'Missingness (and whether it is informative: chi-square / Mann-Whitney of the target between missing and present rows), exact and key duplicates, conflicting labels on identical inputs, spelling variants (case/space/punctuation-normalised collisions), domain-impossible values, robust-z (MAD) outliers split into heavy tails vs glitches, 1:1 redundant encodings, |r|>0.95 pairs, and exact arithmetic relations found by robust (inlier-share) linear and log-linear fits.'],
    ['Leakage', 'Single-column predictive power with a small tree under CV; exact-formula detection for the target; post-outcome naming cues; your declared “not known at prediction time” columns. Suspected leaks are removed before every downstream analysis.'],
    ['Signal', 'Label-permutation test: gradient boosting is cross-validated on the real labels and on 10 shuffled copies. Skill = (score − chance) / (perfect − chance).'],
    ['Relevance', 'Grouped permutation importance on a held-out 30% (all one-hot columns of a feature shuffled together).'],
    ['Validation realism', 'Random vs forward-in-time (TimeSeriesSplit) vs grouped (GroupKFold) CV on the same model; target drift across 8 time periods; adversarial validation (can a model tell old rows from new?).'],
    ['Bake-off', `Six model families with the same preprocessing and folds; OOF predictions give every metric; fit time, per-row latency and pickled size measured. Choice uses the one-standard-error rule plus your interpretability and deployment constraints.`],
    ['Threshold', 'For binary targets, precision / recall / F-beta across thresholds on out-of-fold probabilities; β follows your goal (recall → 2, precision → 0.5).'],
    ['Data size', 'Learning curve from 10% to 100% of training folds, extrapolated with an inverse power law err(n) = a + b·n^(−c), at most 4× beyond measured sizes.'],
    ['Training forecast', 'Real staged loss curves (boosting rounds, MLP epochs) on a 75/25 split; best iteration, generalisation gap and post-minimum rise are read automatically. Time: fit at three sizes, fit t ∝ n^b, extrapolate.'],
    ['Labels', 'Simplified confident learning (per-class confidence thresholds on OOF probabilities); for regression, OOF residuals beyond 6 robust SDs.'],
    ['Scoring', 'Each fitness dimension is 100·exp(−penalty/90) with 45 / 20 / 7 penalty points per critical / major / minor finding; the overall score is a weighted average, capped at 30 if there is no signal and 60 if the raw data leaks. It is a heuristic summary; the findings are the substance.'],
  ];
  const t = r.timings;
  return section('method', n, 'Methodology', `How every number in this report was produced. Total run time ${r.elapsed} s.`,
    `<div class="tbl-wrap"><table class="t"><tbody>${items.map(([a, b]) => `<tr><td style="width:180px"><b>${a}</b></td><td class="small">${E(b)}</td></tr>`).join('')}</tbody></table></div>
    <p class="small muted" style="margin-top:10px">Stage timings: ${Object.entries(t).map(([k, v]) => `${k} ${v}s`).join(' · ')}. Limits: tabular data only; estimates come from this dataset and carry its sampling noise; the tool cannot know your domain, so “check timing of X” prompts need a human answer.</p>`);
}

loadDemos();
// reopen a finished report after a refresh (#job=<id>)
async function openFromHash() {
  const m = location.hash.match(/job=(\w+)/);
  if (!m || (S.r && m[1] === S.job)) return;
  try { const s = await api('/api/job/' + m[1]); if (s.status === 'done') { S.job = m[1]; S.r = s.result; renderReport(); go('report'); } } catch { }
}
openFromHash();
window.addEventListener('hashchange', openFromHash);
