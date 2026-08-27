'use strict';

// One solid glyph per piece type, coloured by side.  The hollow white
// characters (♔♕♖) all but vanish over a light crop, and telling ♔ from ♚ at
// thirty pixels is exactly the judgement the reviewer should not have to make.
const GLYPHS = {
  '.': '·', K: '♚', Q: '♛', R: '♜', B: '♝', N: '♞', P: '♟',
  k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟',
};
const NAMES = {
  '.': 'empty', K: 'white king', Q: 'white queen', R: 'white rook', B: 'white bishop',
  N: 'white knight', P: 'white pawn', k: 'black king', q: 'black queen', r: 'black rook',
  b: 'black bishop', n: 'black knight', p: 'black pawn',
};
const PALETTE = ['K', 'k', 'Q', 'q', 'R', 'r', 'B', 'b', 'N', 'n', 'P', 'p', '.'];

const $ = (sel) => document.querySelector(sel);
const state = {
  id: null, cells: [], labels: [], predicted: [], confidence: [],
  orientation: 'white', side: 'w', cursor: 0, dirty: false, cropOpacity: 0.5,
  nextId: null, prevId: null, threshold: 0.9, saving: false,
  // Squares you have already looked at.  A square the model doubted stays
  // doubted in the record, but once you have ruled on it the cursor should
  // stop coming back to it.
  settled: new Set(),
};

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function toast(message, ms = 1800) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), ms);
}

// -- rendering ---------------------------------------------------------

function buildBoard() {
  const board = $('#board');
  board.innerHTML = '';
  for (let index = 0; index < 64; index++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.index = index;
    cell.innerHTML = '<img alt=""><span class="glyph"></span><span class="conf"></span>';
    cell.onclick = () => { state.cursor = index; paint(); };
    board.appendChild(cell);
  }
}

function paint() {
  const cells = $('#board').children;
  for (let index = 0; index < 64; index++) {
    const cell = cells[index];
    const label = state.labels[index];
    const confidence = state.confidence[index];
    const source = state.cells[index];

    cell.querySelector('img').src = source?.image || '';
    const glyph = cell.querySelector('.glyph');
    glyph.textContent = GLYPHS[label] ?? '?';
    glyph.className = 'glyph ' + (label === '.' ? '' : label === label.toUpperCase() ? 'white' : 'black');
    const conf = cell.querySelector('.conf');
    conf.textContent = confidence === null || confidence === undefined
      ? '' : Math.round(confidence * 100);

    cell.classList.toggle('empty', label === '.');
    cell.classList.toggle('cursor', index === state.cursor);
    cell.classList.toggle('edited', label !== state.predicted[index]);
    const low = confidence !== null && confidence !== undefined && confidence < state.threshold;
    cell.classList.toggle('low', low);
    cell.classList.toggle('verylow', low && confidence < 0.5);
    cell.title = `${squareName(index)} · ${NAMES[label]}` +
      (confidence != null ? ` · model ${Math.round(confidence * 100)}% sure of ${NAMES[state.predicted[index]]}` : '');
  }
  $('#orientation-label').textContent =
    state.orientation === 'white' ? 'white at the bottom' : 'black at the bottom';
  $('#side-label').textContent = state.side === 'w' ? 'white' : 'black';
  paintCoordinates();
  refreshFen();
}

function squareName(index) {
  const row = Math.floor(index / 8), col = index % 8;
  return state.orientation === 'white'
    ? 'abcdefgh'[col] + (8 - row)
    : 'abcdefgh'[7 - col] + (row + 1);
}

function paintCoordinates() {
  const files = state.orientation === 'white'
    ? ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    : ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'];
  const html = files.map((f) => `<span>${f}</span>`).join('');
  $('#files-top').innerHTML = html;
  $('#files-bottom').innerHTML = html;
}

function buildPalette() {
  const palette = $('#palette');
  palette.innerHTML = '';
  for (const label of PALETTE) {
    const button = document.createElement('button');
    const key = label === '.' ? 'space' : label;
    button.innerHTML =
      `<span class="g">${GLYPHS[label]}</span><span>${NAMES[label]}</span><span class="k">${key}</span>`;
    button.onclick = () => setLabel(state.cursor, label);
    palette.appendChild(button);
  }
}

let fenRequest = 0;

async function refreshFen() {
  // Every keystroke repaints, and the replies can come back out of order;
  // only the newest request is allowed to write to the panel.
  const ticket = ++fenRequest;
  try {
    const data = await json('/api/fen', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ labels: state.labels, orientation: state.orientation, side_to_move: state.side }),
    });
    if (ticket !== fenRequest) return;
    $('#fen').textContent = data.fen;
    $('#lichess').href = data.lichess;
    $('#lichess-editor').href = data.lichess_editor;
    const problems = $('#problems');
    problems.innerHTML = data.problems.map((p) => `<li>${p}</li>`).join('');
  } catch (error) {
    if (ticket === fenRequest) $('#fen').textContent = error.message;
  }
}

// -- editing -----------------------------------------------------------

function setLabel(index, label) {
  state.labels[index] = label;
  state.settled.add(index);
  state.dirty = true;
  const next = nextDoubt(index);
  state.cursor = next === null ? Math.min(63, index + 1) : next;
  paint();
}

function nextDoubt(from) {
  for (let step = 1; step <= 64; step++) {
    const index = (from + step) % 64;
    if (state.settled.has(index)) continue;
    const confidence = state.confidence[index];
    if (confidence !== null && confidence !== undefined && confidence < state.threshold) return index;
  }
  return null;
}

function move(dx, dy) {
  let row = Math.floor(state.cursor / 8) + dy;
  let col = (state.cursor % 8) + dx;
  row = Math.max(0, Math.min(7, row));
  col = Math.max(0, Math.min(7, col));
  state.cursor = row * 8 + col;
  paint();
}

// -- loading and saving ------------------------------------------------

async function load(id) {
  const data = await json(`/api/diagram/${id}`);
  state.id = id;
  state.cells = data.cells;
  state.labels = data.cells.map((c) => c.label);
  state.predicted = data.cells.map((c) => c.predicted);
  state.confidence = data.cells.map((c) => c.confidence);
  state.orientation = data.diagram.orientation;
  state.side = data.diagram.side_to_move;
  state.nextId = data.next_id;
  state.prevId = data.prev_id;
  state.dirty = false;
  state.settled = new Set();

  const first = nextDoubt(63);
  state.cursor = first === null ? 0 : first;

  $('#crop').src = data.crop + `?v=${Date.now()}`;
  $('#page').src = data.page_image;
  $('#caption').textContent = data.diagram.caption.split('\n').slice(0, 2).join(' · ') || '(no caption printed)';
  $('#where').textContent = `diagram ${id} · page ${data.diagram.page + 1} · ${data.diagram.status}`;
  $('#remaining').textContent = `${data.remaining} in this queue`;
  $('#prev').disabled = !state.prevId;
  $('#next').disabled = !state.nextId;

  const meta = data.diagram.detect_meta || {};
  $('#detect-pill').textContent =
    `detected ${data.diagram.score.toFixed(2)} · ${meta.occupied_cells ?? '?'} men`;
  const worst = data.diagram.min_confidence;
  const pill = $('#conf-pill');
  pill.textContent = worst == null ? 'not read yet' : `worst square ${Math.round(worst * 100)}%`;
  pill.className = 'pill ' + (worst == null ? '' : worst < 0.5 ? 'bad' : worst < 0.9 ? 'warn' : 'ok');

  paint();
  history.replaceState(null, '', `/review?id=${id}`);
}

async function save(goOn = true) {
  if (state.saving) return;
  state.saving = true;
  try {
    const data = await json(`/api/diagram/${state.id}/save`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ labels: state.labels, orientation: state.orientation, side_to_move: state.side }),
    });
    state.dirty = false;
    const corrected = data.corrections;
    toast(corrected === 0
      ? 'Saved — the model had every square right.'
      : `Saved with ${corrected} correction${corrected === 1 ? '' : 's'}.`);
    if (goOn && state.nextId) {
      await load(state.nextId);
    } else if (goOn) {
      toast('That was the last one in this queue.');
    }
  } catch (error) {
    toast(error.message, 4000);
  } finally {
    state.saving = false;
  }
}

async function reread() {
  try {
    const data = await json(`/api/diagram/${state.id}/reread`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ use_exemplars: true }),
    });
    state.labels = data.labels.slice();
    state.predicted = data.labels.slice();
    state.confidence = data.confidence.slice();
    state.orientation = data.orientation;
    state.side = data.side_to_move;
    state.settled = new Set();
    const first = nextDoubt(63);
    state.cursor = first === null ? 0 : first;
    paint();
    toast(`Read again with ${data.source}${data.exemplars ? ` · ${data.exemplars} exemplars from this book` : ''}.`);
  } catch (error) {
    toast(error.message, 4000);
  }
}

// -- keyboard ----------------------------------------------------------

const PIECE_KEYS = new Set(['k', 'q', 'r', 'b', 'n', 'p']);
const EMPTY_KEYS = new Set([' ', '.', '0', 'Delete', 'Backspace']);

document.addEventListener('keydown', (event) => {
  const tag = event.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;

  const key = event.key;
  const lower = key.toLowerCase();

  // Pieces first, so that the shortcut letters below can never shadow one.
  if (PIECE_KEYS.has(lower)) {
    event.preventDefault();
    // FEN's own convention: a capital letter is a white piece.
    const white = key !== lower;
    return setLabel(state.cursor, white ? lower.toUpperCase() : lower);
  }
  if (EMPTY_KEYS.has(key)) {
    event.preventDefault();
    return setLabel(state.cursor, '.');
  }

  switch (key) {
    case 'ArrowUp': event.preventDefault(); return move(0, -1);
    case 'ArrowDown': event.preventDefault(); return move(0, 1);
    case 'ArrowLeft': event.preventDefault(); return move(-1, 0);
    case 'ArrowRight': event.preventDefault(); return move(1, 0);
    case 'Tab': {
      event.preventDefault();
      const next = nextDoubt(state.cursor);
      if (next === null) return toast('Nothing left below the flag threshold.');
      state.cursor = next;
      return paint();
    }
    case 'Enter': event.preventDefault(); return void save(true);
    case 'Escape': location.href = '/'; return;
    case 'f': event.preventDefault(); return flip();
    case 't': event.preventDefault(); return toggleSide();
    case 'l': event.preventDefault(); return void window.open($('#lichess').href, '_blank', 'noopener');
    case 'a': event.preventDefault(); return void reread();
    case 'c': event.preventDefault(); return cycleCrops();
    default: return;
  }
});

function cycleCrops() {
  // Three useful views: the reading over the picture, the reading alone (does
  // the position make sense?), and the picture alone (what does it really say?).
  const steps = [0.5, 0.0, 1.0];
  const next = steps[(steps.indexOf(state.cropOpacity) + 1) % steps.length];
  state.cropOpacity = next;
  document.documentElement.style.setProperty('--crop-opacity', String(next));
  toast(next === 0 ? 'Showing the reading only.' : next === 1 ? 'Showing the crops only.' : 'Showing both.');
}

function flip() {
  state.orientation = state.orientation === 'white' ? 'black' : 'white';
  state.dirty = true;
  paint();
}

function toggleSide() {
  state.side = state.side === 'w' ? 'b' : 'w';
  state.dirty = true;
  paint();
}

// -- wiring ------------------------------------------------------------

$('#flip').onclick = flip;
$('#side').onclick = toggleSide;
$('#reread').onclick = reread;
$('#save').onclick = () => save(true);
$('#next').onclick = () => state.nextId && load(state.nextId);
$('#prev').onclick = () => state.prevId && load(state.prevId);
$('#copy').onclick = async () => {
  await navigator.clipboard.writeText($('#fen').textContent);
  toast('FEN copied.');
};
$('#reject').onclick = async () => {
  await json(`/api/diagram/${state.id}/status`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ status: 'rejected' }),
  });
  toast('Marked as not a diagram.');
  if (state.nextId) load(state.nextId);
};
$('#threshold').oninput = (event) => {
  state.threshold = Number(event.target.value) / 100;
  $('#threshold-label').textContent = `${event.target.value}%`;
  paint();
};

window.addEventListener('beforeunload', (event) => {
  if (state.dirty) { event.preventDefault(); event.returnValue = ''; }
});

buildBoard();
buildPalette();
const id = new URLSearchParams(location.search).get('id');
if (id) {
  load(Number(id)).catch((error) => toast(error.message, 6000));
} else {
  json('/api/queue?status=pending&order=uncertain&limit=1')
    .then((data) => data.diagrams.length ? load(data.diagrams[0].id) : toast('Nothing to review.'))
    .catch((error) => toast(error.message, 6000));
}
