'use strict';

const $ = (sel) => document.querySelector(sel);
let bookFilter = null;
let watching = null;      // the id of the job this page is following

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.parse(detail).detail ?? detail; } catch { /* plain text */ }
    throw new Error(detail || `${response.status}`);
  }
  return response.json();
}

function toast(message, ms = 2600) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), ms);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function confidencePill(value) {
  if (value === null || value === undefined) return '<span class="pill">not read</span>';
  const cls = value < 0.5 ? 'bad' : value < 0.9 ? 'warn' : 'ok';
  return `<span class="pill ${cls}">${(value * 100).toFixed(0)}% worst square</span>`;
}

// -- opening a book ------------------------------------------------------

async function openBook(file) {
  if (!file) return;
  if (!/\.pdf$/i.test(file.name)) return toast('That is not a PDF.');
  const params = new URLSearchParams({ name: file.name, dpi: $('#dpi').value });
  const pages = $('#pages').value.trim();
  if (pages) params.set('pages', pages);
  try {
    // The body is the file itself: no form encoding to agree on, and the
    // browser streams it rather than holding a second copy in memory.
    const data = await json('/api/books?' + params, { method: 'POST', body: file });
    follow(data.job);
  } catch (error) {
    toast(error.message, 6000);
  }
}

async function start(url, label) {
  try {
    follow((await json(url, { method: 'POST' })).job);
  } catch (error) {
    toast(`${label}: ${error.message}`, 6000);
  }
}

// -- watching the work ---------------------------------------------------

const TITLES = { ingest: 'Reading', train: 'Training', reread: 'Re-reading', pieces: 'Downloading' };

function follow(job) {
  watching = job.id;
  paintJob(job);
  poll();
}

async function poll() {
  if (!watching) return;
  let job;
  try {
    job = await json(`/api/jobs/${watching}`);
  } catch {
    return setTimeout(poll, 2000);      // a restart, or a slow moment
  }
  paintJob(job);
  if (job.state === 'running' || job.state === 'queued') return setTimeout(poll, 600);

  watching = null;
  await loadStats().catch(() => {});
  await loadQueue().catch(() => {});
  if (job.state === 'done' && job.kind === 'ingest' && job.result.book_id) {
    // What the reader came here for: the diagrams, not a report about them.
    if (job.found > 0) {
      location.href = `/review?book=${job.result.book_id}`;
    } else {
      toast('No diagrams found in that book. If it is a scan, try “finest” under Options.', 8000);
    }
  }
}

function paintJob(job) {
  const card = $('#job');
  card.hidden = false;
  card.classList.toggle('failed', job.state === 'failed');
  $('#job-title').textContent = `${TITLES[job.kind] ?? 'Working'} ${job.label}`;
  $('#job-count').textContent = job.total ? `${job.done} / ${job.total}` : '';

  const bar = $('#job-bar');
  bar.style.width = `${(job.state === 'done' ? 1 : job.fraction) * 100}%`;
  bar.classList.toggle('indeterminate', !job.total && job.state === 'running');

  const actions = $('#job-actions');
  actions.innerHTML = '';
  if (job.state === 'failed') {
    $('#job-note').textContent = job.error;
    addAction(actions, 'Dismiss', () => { card.hidden = true; });
  } else if (job.state === 'done') {
    $('#job-note').textContent = job.result.summary || 'Done.';
    if (job.result.book_id) {
      addAction(actions, 'Review these diagrams',
                () => { location.href = `/review?book=${job.result.book_id}`; }, true);
    }
    addAction(actions, 'Dismiss', () => { card.hidden = true; });
  } else {
    $('#job-note').textContent = job.note || '';
    if (job.found) $('#job-note').textContent += ` · ${job.found} diagram(s) so far`;
  }
}

function addAction(parent, label, onClick, primary = false) {
  const button = document.createElement('button');
  button.textContent = label;
  if (primary) button.className = 'primary';
  button.onclick = onClick;
  parent.appendChild(button);
}

// -- the lists -----------------------------------------------------------

async function loadStats() {
  const data = await json('/api/stats');
  const s = data.stats;
  $('#summary').textContent =
    `${s.books} book${s.books === 1 ? '' : 's'} · ${s.diagrams} diagrams · ` +
    `${s.verified} verified · ${s.pending} to review`;
  $('#model').textContent = data.model
    ? `model #${data.model.id}`
    : 'using the packaged model';
  $('#train-note').textContent = s.labelled_squares
    ? `${s.labelled_squares} labelled squares to train on`
    : 'nothing corrected yet — training would learn only from generated diagrams';

  // Nothing to say about a library that is empty: on a first run the page is
  // the drop zone and three sentences, not two headed tables with no rows.
  const any = data.books.length > 0;
  $('#library').hidden = !any;
  $('#teach').hidden = !any;
  $('#steps').hidden = any;

  const body = $('#books tbody');
  body.innerHTML = '';
  for (const book of data.books) {
    const done = book.diagram_count ? book.verified_count / book.diagram_count : 0;
    const tr = document.createElement('tr');
    tr.dataset.bookId = book.id;
    tr.innerHTML = `
      <td class="clickable">
        <div class="title" title="${escapeHtml(book.title)}">${escapeHtml(book.title)}</div>
        <div class="muted" style="font-size:12px">${book.pages} pages · ${book.diagram_count} diagrams
          · ${book.verified_count} checked</div>
        <div class="bar" style="margin-top:6px"><i style="width:${(done * 100).toFixed(1)}%"></i></div>
      </td>
      <td style="width:1%;white-space:nowrap">
        <button class="primary review-book">Review</button>
        <button class="quiet forget-book" title="remove this book and its diagrams">✕</button>
      </td>`;
    tr.querySelector('.clickable').onclick = () => {
      bookFilter = bookFilter === book.id ? null : book.id;
      loadQueue();
      highlightBook();
    };
    tr.querySelector('.review-book').onclick = () => {
      location.href = `/review?book=${book.id}`;
    };
    tr.querySelector('.forget-book').onclick = () => forget(book);
    body.appendChild(tr);
  }
  highlightBook();
}

async function forget(book) {
  const verified = book.verified_count
    ? `\n\n${book.verified_count} verified diagram(s) will go with it, and those are training data.`
    : '';
  if (!confirm(`Remove “${book.title}” and its ${book.diagram_count} diagrams?${verified}`)) return;
  try {
    await json(`/api/books/${book.id}`, { method: 'DELETE' });
    if (bookFilter === book.id) bookFilter = null;
    await loadStats();
    await loadQueue();
    toast('Removed.');
  } catch (error) {
    toast(error.message, 6000);
  }
}

function highlightBook() {
  for (const tr of document.querySelectorAll('#books tbody tr')) {
    tr.classList.toggle('selected', Number(tr.dataset.bookId) === bookFilter);
  }
  $('#review-all').hidden = !bookFilter;
}

async function loadQueue() {
  const params = new URLSearchParams({
    status: $('#status').value, order: $('#order').value, limit: '300',
  });
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

// -- wiring --------------------------------------------------------------

$('#choose').onclick = () => $('#file').click();
$('#file').onchange = (event) => {
  openBook(event.target.files[0]);
  event.target.value = '';        // so the same file can be opened twice
};
$('#demo').onclick = () => start('/api/books/demo', 'sample book');
$('#train').onclick = () => {
  if (confirm('Training takes 20-45 minutes. Leave this window open; you can '
            + 'keep reviewing in another tab. Start?')) start('/api/train', 'training');
};
$('#reread').onclick = () => start('/api/reread' + (bookFilter ? `?book_id=${bookFilter}` : ''), 're-reading');
$('#fetch').onclick = () => start('/api/pieces/fetch', 'downloading styles');
$('#advanced-toggle').onclick = () => { $('#advanced').hidden = !$('#advanced').hidden; };
$('#review-all').onclick = () => { if (bookFilter) location.href = `/review?book=${bookFilter}`; };
$('#status').onchange = loadQueue;
$('#order').onchange = loadQueue;

const drop = $('#drop');
for (const name of ['dragenter', 'dragover']) {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add('over');
  });
}
for (const name of ['dragleave', 'drop']) {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    if (name === 'dragleave' && drop.contains(event.relatedTarget)) return;
    drop.classList.remove('over');
  });
}
drop.addEventListener('drop', (event) => openBook(event.dataTransfer.files[0]));
// Dropping a PDF anywhere else would otherwise navigate away from the app.
for (const name of ['dragover', 'drop']) {
  document.addEventListener(name, (event) => {
    if (!drop.contains(event.target)) event.preventDefault();
  });
}

loadStats()
  .then(loadQueue)
  .then(() => json('/api/jobs'))
  .then((data) => { if (data.active) follow(data.active); })
  .catch((error) => { $('#summary').textContent = error.message; });
