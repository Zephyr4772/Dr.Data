// Tiny SVG chart kit. Every function returns an SVG string sized by viewBox (scales to its container).
// Colours come from CSS variables so light/dark themes just work.
const Charts = (() => {
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const num = (v, d = 2) => {
    if (v == null || !isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(0) + 'k';
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(Math.max(0, d - 2));
    if (a >= 10) return v.toFixed(Math.max(0, d - 1));
    return v.toFixed(d);
  };
  const lin = (d0, d1, r0, r1) => v => r0 + (d1 === d0 ? 0.5 : (v - d0) / (d1 - d0)) * (r1 - r0);
  const ticks = (lo, hi, n = 5) => {
    if (!isFinite(lo) || !isFinite(hi)) return [];
    if (hi === lo) return [lo];
    const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(10));
    return out;
  };
  const svg = (w, h, body, cls = '') => `<svg class="chart ${cls}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" role="img">${body}</svg>`;

  // ---------------------------------------------------------------- horizontal bars
  function hbar({ labels, values, err = null, fmt = v => num(v, 3), max = null, min = null, cls = null, ref = null, refLabel = '', width = 640, unit = '' }) {
    const n = labels.length, rowH = 26, top = 8, lw = Math.min(210, 12 + 7 * Math.max(...labels.map(l => String(l).length))), right = 64;
    const h = top + n * rowH + 22;
    const lo = min ?? Math.min(0, ...values), hi = max ?? Math.max(...values.map((v, i) => v + (err ? err[i] : 0)), ref ?? -Infinity, 1e-9);
    const x = lin(lo, hi, lw, width - right);
    let b = '';
    ticks(lo, hi, 4).forEach(t => { b += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${top + n * rowH}"/><text class="tick" x="${x(t)}" y="${top + n * rowH + 15}" text-anchor="middle">${num(t)}</text>`; });
    labels.forEach((l, i) => {
      const y = top + i * rowH, v = values[i], c = cls ? cls(i, v) : 'bar';
      const x0 = x(Math.max(lo, Math.min(0, v))), x1 = x(Math.max(0, v));
      b += `<g><title>${esc(l)}: ${fmt(v)}${unit ? ' ' + unit : ''}</title>
        <text class="lbl" x="${lw - 8}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(String(l).length > 30 ? String(l).slice(0, 29) + '…' : l)}</text>
        <rect class="${c}" x="${Math.min(x0, x1)}" y="${y + 5}" width="${Math.max(1.5, Math.abs(x1 - x0))}" height="${rowH - 10}" rx="3"/>
        ${err ? `<line class="err" x1="${x(v - err[i])}" x2="${x(v + err[i])}" y1="${y + rowH / 2}" y2="${y + rowH / 2}"/>` : ''}
        <text class="val" x="${Math.max(x1, x0) + 6 + (err ? Math.max(0, x(v + err[i]) - x1) : 0)}" y="${y + rowH / 2 + 4}">${fmt(v)}</text></g>`;
    });
    if (ref != null) b += `<line class="ref" x1="${x(ref)}" x2="${x(ref)}" y1="${top - 4}" y2="${top + n * rowH}"/><text class="reflbl" x="${x(ref) + 4}" y="${top + 6}">${esc(refLabel)}</text>`;
    return svg(width, h, b);
  }

  // ---------------------------------------------------------------- vertical bars (few categories)
  function vbar({ labels, values, fmt = v => num(v, 0), cls = null, width = 520, height = 220, ylabel = '' }) {
    const n = labels.length, left = 46, bottom = 40, top = 14;
    const hi = Math.max(...values) * 1.12 || 1;
    const y = lin(0, hi, height - bottom, top);
    const bw = (width - left - 10) / n;
    let b = '';
    ticks(0, hi, 4).forEach(t => { b += `<line class="grid" x1="${left}" x2="${width - 6}" y1="${y(t)}" y2="${y(t)}"/><text class="tick" x="${left - 6}" y="${y(t) + 4}" text-anchor="end">${num(t)}</text>`; });
    labels.forEach((l, i) => {
      const x = left + i * bw + bw * .15, w = bw * .7, v = values[i];
      b += `<g><title>${esc(l)}: ${fmt(v)}</title><rect class="${cls ? cls(i) : 'bar'}" x="${x}" y="${y(v)}" width="${w}" height="${y(0) - y(v)}" rx="3"/>
        <text class="val" x="${x + w / 2}" y="${y(v) - 5}" text-anchor="middle">${fmt(v)}</text>
        <text class="lbl" x="${x + w / 2}" y="${height - bottom + 16}" text-anchor="middle">${esc(String(l).slice(0, 14))}</text></g>`;
    });
    if (ylabel) b += `<text class="axlbl" transform="translate(12 ${(height - bottom + top) / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
    return svg(width, height, b);
  }

  // ---------------------------------------------------------------- histogram from counts + [lo, hi]
  function hist({ counts, edges, width = 520, height = 180, xlabel = '', marks = [] }) {
    const left = 40, bottom = 34, top = 10, n = counts.length;
    const hi = Math.max(...counts) * 1.1 || 1;
    const x = lin(edges[0], edges[1], left, width - 8), y = lin(0, hi, height - bottom, top);
    const bw = (width - 8 - left) / n;
    let b = '';
    ticks(0, hi, 3).forEach(t => { b += `<line class="grid" x1="${left}" x2="${width - 8}" y1="${y(t)}" y2="${y(t)}"/><text class="tick" x="${left - 6}" y="${y(t) + 4}" text-anchor="end">${num(t, 0)}</text>`; });
    counts.forEach((c, i) => { b += `<rect class="bar" x="${left + i * bw + .5}" y="${y(c)}" width="${bw - 1}" height="${y(0) - y(c)}"><title>${c}</title></rect>`; });
    ticks(edges[0], edges[1], 5).forEach(t => { b += `<text class="tick" x="${x(t)}" y="${height - bottom + 15}" text-anchor="middle">${num(t)}</text>`; });
    marks.forEach(m => { b += `<line class="ref" x1="${x(m.x)}" x2="${x(m.x)}" y1="${top}" y2="${height - bottom}"/><text class="reflbl" x="${x(m.x) + 4}" y="${top + 10}">${esc(m.label)}</text>`; });
    if (xlabel) b += `<text class="axlbl" x="${(left + width) / 2}" y="${height - 4}" text-anchor="middle">${esc(xlabel)}</text>`;
    return svg(width, height, b);
  }

  // ---------------------------------------------------------------- line chart
  // series: [{name, x:[], y:[], cls:'s1', dash:false, band:[lo[],hi[]], dots:false}]
  function line({ series, width = 640, height = 260, xlabel = '', ylabel = '', vlines = [], hlines = [], ylo = null, yhi = null, logx = false, yfmt = v => num(v, 2), xfmt = v => num(v, 0), legend = true, xcat = null }) {
    const left = 52, right = 14, top = legend ? 30 : 12, bottom = 40;
    const xs = series.flatMap(s => s.x), ys = series.flatMap(s => [...s.y, ...(s.band ? [...s.band[0], ...s.band[1]] : [])]).filter(v => v != null && isFinite(v));
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = ylo ?? Math.min(...ys), y1 = yhi ?? Math.max(...ys);
    hlines.forEach(h => { y0 = Math.min(y0, h.y); y1 = Math.max(y1, h.y); });
    const pad = (y1 - y0) * .08 || Math.abs(y1) * .1 || 1;
    if (ylo == null) y0 -= pad; if (yhi == null) y1 += pad;
    const tx = logx ? v => Math.log10(Math.max(v, 1e-9)) : v => v;
    const x = lin(tx(x0), tx(x1), left, width - right), X = v => x(tx(v)), y = lin(y0, y1, height - bottom, top);
    let b = '';
    ticks(y0, y1, 4).forEach(t => { b += `<line class="grid" x1="${left}" x2="${width - right}" y1="${y(t)}" y2="${y(t)}"/><text class="tick" x="${left - 6}" y="${y(t) + 4}" text-anchor="end">${yfmt(t)}</text>`; });
    if (xcat) xcat.forEach((l, i) => { if (i % Math.ceil(xcat.length / 8) === 0) b += `<text class="tick" x="${X(i)}" y="${height - bottom + 16}" text-anchor="middle">${esc(l)}</text>`; });
    else {
      const xt = logx ? [x0, ...[1e1, 1e2, 1e3, 1e4, 1e5, 1e6].filter(v => v > x0 && v < x1), x1] : ticks(x0, x1, 6);
      xt.forEach(t => { b += `<text class="tick" x="${X(t)}" y="${height - bottom + 16}" text-anchor="middle">${xfmt(t)}</text>`; });
    }
    hlines.forEach(h => { b += `<line class="ref ${h.cls || ''}" x1="${left}" x2="${width - right}" y1="${y(h.y)}" y2="${y(h.y)}"/><text class="reflbl" x="${width - right - 4}" y="${y(h.y) - 5}" text-anchor="end">${esc(h.label || '')}</text>`; });
    series.forEach(s => {
      if (s.band) {
        const up = s.x.map((v, i) => `${X(v)},${y(s.band[1][i])}`), dn = s.x.map((v, i) => `${X(v)},${y(s.band[0][i])}`).reverse();
        b += `<polygon class="band ${s.cls}" points="${up.concat(dn).join(' ')}"/>`;
      }
      const pts = s.x.map((v, i) => s.y[i] == null || !isFinite(s.y[i]) ? null : `${X(v)},${y(s.y[i])}`).filter(Boolean);
      b += `<polyline class="ln ${s.cls} ${s.dash ? 'dash' : ''}" points="${pts.join(' ')}"/>`;
      if (s.dots) s.x.forEach((v, i) => { if (s.y[i] != null && isFinite(s.y[i])) b += `<circle class="dot ${s.cls}" cx="${X(v)}" cy="${y(s.y[i])}" r="3.2"><title>${esc(s.name)}: ${yfmt(s.y[i])} @ ${xcat ? xcat[v] : xfmt(v)}</title></circle>`; });
    });
    vlines.forEach(v => { b += `<line class="ref ${v.cls || ''}" x1="${X(v.x)}" x2="${X(v.x)}" y1="${top}" y2="${height - bottom}"/><text class="reflbl ${v.cls || ''}" x="${X(v.x) + 5}" y="${top + 12}">${esc(v.label || '')}</text>`; });
    if (legend) {
      let lx = left;
      series.filter(s => s.name).forEach(s => { b += `<line class="ln ${s.cls} ${s.dash ? 'dash' : ''}" x1="${lx}" x2="${lx + 18}" y1="12" y2="12"/><text class="leg" x="${lx + 23}" y="16">${esc(s.name)}</text>`; lx += 34 + s.name.length * 6.6; });
    }
    if (xlabel) b += `<text class="axlbl" x="${(left + width) / 2}" y="${height - 6}" text-anchor="middle">${esc(xlabel)}</text>`;
    if (ylabel) b += `<text class="axlbl" transform="translate(12 ${(height - bottom + top) / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
    return svg(width, height, b);
  }

  // ---------------------------------------------------------------- radar
  function radar({ axes, size = 340 }) {
    const c = size / 2, r = size / 2 - 78, n = axes.length;
    const pt = (i, v) => { const a = -Math.PI / 2 + i * 2 * Math.PI / n; return [c + Math.cos(a) * r * v, c + Math.sin(a) * r * v]; };
    let b = '';
    [.25, .5, .75, 1].forEach(k => { b += `<polygon class="rgrid" points="${axes.map((_, i) => pt(i, k).join(',')).join(' ')}"/>`; });
    axes.forEach((a, i) => {
      const [x, y] = pt(i, 1), [lx, ly] = pt(i, 1.2);
      b += `<line class="rgrid" x1="${c}" y1="${c}" x2="${x}" y2="${y}"/>`;
      const anchor = Math.abs(lx - c) < 8 ? 'middle' : lx > c ? 'start' : 'end';
      b += `<text class="rlbl" x="${lx}" y="${ly}" text-anchor="${anchor}">${esc(a.label)}</text><text class="rval ${a.tone}" x="${lx}" y="${ly + 14}" text-anchor="${anchor}">${Math.round(a.value * 100)}</text>`;
    });
    b += `<polygon class="rshape" points="${axes.map((a, i) => pt(i, Math.max(.03, a.value)).join(',')).join(' ')}"/>`;
    axes.forEach((a, i) => { const [x, y] = pt(i, Math.max(.03, a.value)); b += `<circle class="rdot ${a.tone}" cx="${x}" cy="${y}" r="4.5"><title>${esc(a.label)}: ${Math.round(a.value * 100)}/100</title></circle>`; });
    return svg(size, size, b, 'radar');
  }

  // ---------------------------------------------------------------- score ring
  function ring(score, tone, size = 180) {
    const r = size / 2 - 12, C = 2 * Math.PI * r, c = size / 2;
    return svg(size, size, `<circle class="ring-bg" cx="${c}" cy="${c}" r="${r}"/>
      <circle class="ring-fg ${tone}" cx="${c}" cy="${c}" r="${r}" stroke-dasharray="${C * score / 100} ${C}" transform="rotate(-90 ${c} ${c})"/>
      <text class="ring-num" x="${c}" y="${c + 12}" text-anchor="middle">${score}</text>
      <text class="ring-sub" x="${c}" y="${c + 34}" text-anchor="middle">of 100</text>`, 'ring');
  }

  // ---------------------------------------------------------------- strip plot: null distribution vs real value
  function nullDist({ nul, real, label, width = 640, height = 120 }) {
    const vals = [...nul, real];
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * .15 || .05; lo -= pad; hi += pad;
    const x = lin(lo, hi, 30, width - 30);
    let b = `<line class="axis" x1="30" x2="${width - 30}" y1="70" y2="70"/>`;
    ticks(lo, hi, 6).forEach(t => { b += `<text class="tick" x="${x(t)}" y="88" text-anchor="middle">${num(t, 2)}</text>`; });
    nul.forEach((v, i) => { b += `<circle class="nulldot" cx="${x(v)}" cy="${58 - (i % 3) * 9}" r="5"><title>shuffled run ${i + 1}: ${num(v, 3)}</title></circle>`; });
    b += `<line class="realline" x1="${x(real)}" x2="${x(real)}" y1="18" y2="74"/><text class="reallbl" x="${x(real)}" y="13" text-anchor="middle">real labels: ${num(real, 3)}</text>`;
    b += `<text class="axlbl" x="${width / 2}" y="${height - 6}" text-anchor="middle">${esc(label)} — grey dots: same model trained on shuffled labels (pure chance)</text>`;
    return svg(width, height, b);
  }

  // ---------------------------------------------------------------- sparkline bars
  function spark(counts, cls = 'bar') {
    const w = 120, h = 26, n = counts.length, hi = Math.max(...counts) || 1, bw = w / n;
    return svg(w, h, counts.map((c, i) => `<rect class="${cls}" x="${i * bw + .4}" y="${h - (c / hi) * (h - 2)}" width="${Math.max(.8, bw - .8)}" height="${(c / hi) * (h - 2)}"/>`).join(''), 'spark');
  }

  return { hbar, vbar, hist, line, radar, ring, nullDist, spark, num, esc };
})();
