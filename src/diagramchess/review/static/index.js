const $ = (sel) => document.querySelector(sel);
let bookFilter = null;

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function confidencePill(value) {
  if (value === null || value === undefined) return '<span class="pill">not read</span>';
  const percent = (value * 100).toFixed(0);
  const cls = value < 0.5 ? 'bad' : value < 0.9 ? 'warn' : 'ok';
  return `<span class="pill ${cls}">${percent}% worst square</span>`;
}

async function loadStats() {
  const data = await json('/api/stats');
  const s = data.stats;
  $('#summary').textContent =
    `${s.books} book${s.books === 1 ? '' : 's'} · ${s.diagrams} diagrams · ` +
    `${s.verified} verified · ${s.pending} to review · ${s.labelled_squares} labelled squares`;
  $('#model').textContent = data.model
    ? `model #${data.model.id} · ${JSON.parse(data.model.metrics).val_accuracy?.toFixed?.(4) ?? ''}`
    : 'no model registered';

  const body = $('#books tbody');
  body.innerHTML = '';
  for (const book of data.books) {
    const done = book.diagram_count ? book.verified_count / book.diagram_count : 0;
    const tr = document.createElement('tr');
    tr.className = 'clickable';
    tr.innerHTML = `
      <td>
        <div>${escapeHtml(book.title)}</div>
        <div class="muted" style="font-size:12px">${book.pages} pages · ${book.diagram_count} diagrams</div>
        <div class="bar" style="margin-top:6px"><i style="width:${(done * 100).toFixed(1)}%"></i></div>
      </td>
      <td style="width:1%;white-space:nowrap" class="muted">${book.verified_count}/${book.diagram_count}</td>`;
    tr.onclick = () => { bookFilter = bookFilter === book.id ? null : book.id; loadQueue(); highlightBook(); };
    tr.dataset.bookId = book.id;
    body.appendChild(tr);
  }
  highlightBook();
}

function highlightBook() {
  for (const tr of document.querySelectorAll('#books tbody tr')) {
    tr.style.background = Number(tr.dataset.bookId) === bookFilter
      ? 'color-mix(in srgb, var(--accent) 12%, transparent)' : '';
  }
}

async function loadQueue() {
  const params = new URLSearchParams({ status: $('#status').value, order: $('#order').value, limit: '300' });
  if (bookFilter) params.set('book_id', bookFilter);
  const data = await json('/api/queue?' + params);
  const body = $('#queue tbody');
  body.innerHTML = '';
  $('#empty').style.display = data.diagrams.length ? 'none' : '';
  for (const d of data.diagrams) {
    const tr = document.createElement('tr');
    tr.className = 'clickable';
    tr.innerHTML = `
      <td class="mono muted">${d.id}</td>
      <td>${d.page + 1}</td>
      <td>${escapeHtml(d.caption.split('\n')[0].slice(0, 60))}</td>
      <td>${confidencePill(d.min_confidence)}</td>
      <td class="muted mono" style="font-size:12px" title="detection score, and squares carrying ink"
          >${d.score.toFixed(2)} · ${d.detect_meta.occupied_cells ?? '?'}</td>`;
    tr.onclick = () => { location.href = `/review?id=${d.id}`; };
    body.appendChild(tr);
  }
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

$('#status').onchange = loadQueue;
$('#order').onchange = loadQueue;
loadStats().then(loadQueue).catch((e) => { $('#summary').textContent = e.message; });
