# diagramchess

A companion for reading chess books as PDFs. It finds the diagrams, reads the
position off each one, and hands you a Lichess analysis board — so studying a
book stops meaning setting positions up by hand.

It learns as you use it. The first pass is done by a classifier trained on
diagrams the tool drew itself; you check its readings in a keyboard-driven
review screen, and every correction becomes training data. Because a book sets
its diagrams in one figurine font throughout, a handful of verified diagrams is
usually enough to make the rest of that book read cleanly.

```
dgc app                       # opens a window: drop a PDF in, and review what it read
```

Opening a book, watching it being read, checking the diagrams, training on your
corrections and re-reading with the result all happen in that one window.
`dgc install-launcher` puts a shortcut in your applications menu, after which
there is no terminal at all.

The same steps are separate commands if you would rather script them:

```
dgc pieces --fetch            # ~40 figurine styles to train against (worth it, see below)
dgc demo-book sample.pdf      # a generated chess book, if you want to try it first
dgc ingest your-book.pdf      # find the diagrams and read them
dgc review                    # check them; corrections are saved as training data
dgc accuracy                  # is it good enough to stop checking every one?
dgc train && dgc reread       # fold your corrections back in
dgc export --format pgn       # take the positions elsewhere
```

A classifier ships with the package, so there is no training run to sit through
before the first book. On a book set in a figurine font it has never seen it
reads about 99% of squares — under one correction a diagram — and the review
loop takes it lower still. `dgc train` then replaces it with a model that has
seen your corrections.

## Installing

Python 3.10 or newer. If a terminal is unfamiliar territory,
[GETTING-STARTED.md](GETTING-STARTED.md) walks through the same thing from
scratch, one step at a time.

```bash
git clone https://github.com/official-rey/diagramchess
cd diagramchess

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e ".[ml]"
```

That takes a few minutes, most of it PyTorch. Check it worked:

```bash
dgc --version
dgc pieces                           # lists the figurine styles it can draw
```

**Cairo.** Piece artwork is drawn through Cairo, because it is the only
renderer here that handles gradient fills — and getting that wrong is silent
(see *Four measurements that changed the design*). `pip` installs the Python
binding; the native library may need installing separately:

| | |
|---|---|
| Debian/Ubuntu | `sudo apt install libcairo2` |
| macOS | `brew install cairo` |
| Windows | usually already works; if not, `pip install pycairo` |

`dgc pieces` says so plainly if Cairo is not working. It is not fatal — reading
books with the packaged model is unaffected, and the tool falls back to a
renderer that merely drops gradient fills. It only matters if you go on to
download extra styles and train on them.

**PyTorch.** The `ml` extra pulls it in. If you would rather have the smaller
CPU-only build:

```bash
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Everything except training and the neural classifier — finding diagrams, the
review screen, exporting FENs, reading a book from exemplars you have verified —
works with no PyTorch at all.

## Using it on your own PDFs

```bash
cd wherever/you/keep/books
dgc app
```

Drop the PDF on the window that opens. It finds every diagram, reads each one —
a few hundred pages takes a couple of minutes, with a progress bar — and hands
you the review screen when it is done. The same thing from the command line is
`dgc ingest my-chess-book.pdf` followed by `dgc review`.

The queue is ordered by how unsure the model is, so the diagrams most likely to
be wrong come first. Check a few. If they are all correct, you can stop —
`dgc export` already has the positions.

```bash
dgc export --format board            # positions and Lichess links in the terminal
dgc export --format pgn > book.pgn   # one PGN entry per position
dgc export --format csv > book.csv
```

`--status pending` exports what you have not reviewed, `--status all` everything.

A few things worth knowing:

- **Everything lives in `.diagramchess/`** in whatever directory you ran from,
  including your corrections. Move it, back it up, delete it to start over. Use
  `-w path/to/workspace` to keep one workspace for several books.
- **Ingest at the default 200 dpi.** 150 works; below about 120 the diagrams get
  too small to find reliably. Use `--dpi 300` for a book with very small
  diagrams, at the cost of speed.
- **`--pages 40-90`** limits ingest to a range, which is worth doing on a first
  run to see how a book goes before committing to all of it. The window has the
  same thing under *Options*.
- **A book opened through the window is copied into the workspace**, so it can
  redraw pages later; one opened with `dgc ingest` is read where it lies, and
  removing it from the book list never deletes a file you chose yourself.
- **Re-ingesting is safe.** Diagrams are recognised by where they sit on the
  page, so a second pass keeps everything you verified.

### If it gets things wrong

Check `dgc accuracy` first — it compares the model against your own corrections
and tells you what kind of problem you have. Then, from the buttons at the
bottom of the window, or equivalently:

```bash
dgc pieces --fetch                   # ~40 more figurine styles, a few megabytes
dgc train                            # 20-45 min on a laptop CPU
dgc reread                           # re-read the book with the better model
```

`dgc train` uses both the extra styles and every square you have corrected, so
the more of the book you have reviewed the more it has to work with. Twenty
verified diagrams is plenty to make a difference.

If a whole diagram comes out as nonsense, the lattice was cut in the wrong
place: look at the crop in the review screen and check the cell boundaries fall
between the pieces. If the board is upside down press `f`; if the side to move
is wrong press `t`.

## How it works

**Finding the diagrams.** Three independent proposal sources are tried on every
page: the raster images the PDF draws (most books paste each diagram in as one
picture), clusters of vector rectangles (books that draw their boards as
graphics), and connected dark regions and ruled lattices in the rendered page
(everything else, including scans). Each proposal is then verified by fitting
the 8×8 lattice inside it and scoring what the cells contain.

None of this is learned, so it works on the first PDF you feed it. The scoring
does two things worth naming:

- The lattice is found by looking for **nine coherent rules per axis** — lines
  that produce an edge in nearly every row they cross. Summing raw gradient
  instead, as the obvious implementation does, makes the fitter chase knights:
  thirty-two pieces put far more ink on a page than eighteen thin rules do.
- A chessboard and a **tournament crosstable** are the same shape, and chess
  books are full of crosstables. What separates them is the cell contents: a
  diagram has a handful to a boardful of cells carrying a big centred glyph and
  the rest empty, while a crosstable has a small mark in every cell and a
  lattice that keeps going past where a board would end.

**Reading the position.** Each board is cut into 64 padded square crops and
classified 13 ways (12 pieces and empty) by a small convolutional net. Each
crop has its mean subtracted, so how darkly a book prints carries no information
the net can lean on — but its contrast is only scaled down to a floor, never
stretched up to it, because how much contrast a crop has is the strongest
evidence there is for empty against occupied. Dividing that out was a real bug;
see *scanner noise* below.

**Learning your book.** Alongside the net, an **exemplar bank** holds the crops
you have verified in the book you are reading, matched by nearest neighbour.
The net knows what chess pieces look like in general; the bank knows what they
look like *in this book*.

How much say the bank gets is the fiddliest decision in the tool, and it is set
by two things: how much of the label set the bank covers (it can never answer
for a piece type it has not seen), and how far apart the two readers are across
the board — because wide disagreement means the net is reading a figurine style
it does not know, and that is exactly when the bank should win. Below two
verified diagrams the bank is ignored entirely. The numbers behind all of that
are under *Four measurements that changed the design*.

## The window

`dgc app` is the whole tool without a terminal: it starts the server, waits for
it to answer, and only then opens a browser on it — a window that appears
before the server is up shows a connection error and asks the reader to
reload, which is the sort of thing this command exists to prevent. If the port
is taken it takes the next free one rather than failing on someone's desktop.

Opening a book is a PDF posted as a raw request body: no form parser to add as
a dependency, and the browser streams it instead of holding a second copy in
memory. Reading it, training, and re-reading all run as jobs on a single
background worker, one at a time — they are CPU-bound and share a SQLite file,
so a second one running concurrently would only make the first slower. The page
polls the job and, when a book has been read, goes straight to the review
screen for it.

`dgc install-launcher` writes a desktop entry, a small `.app` bundle, or a
`.bat` on the Desktop, depending on the platform, with the interpreter path,
the workspace and the port baked in — a shortcut runs with no shell, no PATH
and no working directory, so nothing may be left to be looked up.
`--remove` takes it back.

Because the server can now import files and start training, and because
anything in the browser can reach localhost, a request that changes something
must carry this app's own origin. Browsers attach `Origin` to every cross-site
write, so the check costs nothing and blocks the obvious mischief; tools that
are not browsers send none and are left alone.

## The review screen

`dgc app` opens it; `dgc review` serves it without opening a window. The queue
is ordered by how unsure the model is, so the diagrams that teach it most come
first. Within a diagram, the cursor starts on the least confident square.

| key | |
|---|---|
| `↑ ↓ ← →` | move the cursor |
| `k q r b n p` | set a black piece |
| `shift` + those | set a white piece (FEN's own convention) |
| `space` or `.` | empty the square |
| `tab` | jump to the next square below the flag threshold |
| `f` | flip the board |
| `t` | switch the side to move |
| `c` | crops / the model's reading / both |
| `a` | read it again, using this book's exemplars |
| `l` | open it on Lichess |
| `enter` | save and go to the next |

Each cell shows the actual crop with the model's reading drawn over it, so
agreement looks clean and disagreement looks doubled. The crop is multiplied
into the square rather than laid on top, which drops the paper and keeps the
ink — the printed diagram appears to be drawn on the board. Squares the model
is unsure about are ringed and carry their confidence; the rest carry nothing,
because a number on every square is sixty-one reassurances hiding the three
readings worth a second look. The threshold is a slider.

The piece colours are fixed rather than taken from the interface theme. That
sounds like a detail and is not: they were once drawn in the theme's ink, so in
dark mode both colours came out near-white and the single thing the screen
exists to show — which side a piece belongs to — was the thing it could not
show. `tests/test_review_ui.py` holds that shut.

Saving records all 64 squares, including the ones the model got right — those
are what teach it that its confident answers are correct.

## Knowing when it is good enough

`dgc accuracy` compares what the model said against what you said, on the
diagrams you have already checked. That is the only number that answers the
question you actually care about. Real output, from a book set in `leipzig` —
a font the model was never trained on, printed small and faintly:

```
measured against 20 diagram(s) you verified (1280 squares)
  squares read correctly:   92.81%
  diagrams read perfectly:  10.00%
  corrections per diagram:   4.60
  above 99% confidence: 1016 squares, 8 of them wrong (0.79%)
  above 95% confidence: 1062 squares, 16 of them wrong (1.51%)
  above 90% confidence: 1084 squares, 20 of them wrong (1.85%)
  mistakes it makes most:
      white pawn read as white bishop: 39
      black pawn read as black bishop: 22
      white rook read as black rook: 10
      white king read as white queen: 9
  -> about 4.6 correction(s) per diagram; keep verifying, and retrain once you
     have twenty or so diagrams done
```

Two things to read off that. The **mistake list** tells you what kind of problem
you have: pawn-for-bishop is the classifier struggling with a small figurine,
while a run of colour flips points at the artwork or the printing rather than
the shapes. The **confidence bands** tell you what a review threshold would
cost: here 0.79% of the squares the model was 99% sure of were still wrong, so
this is not yet a book to spot-check. When that figure reaches zero across a few
thousand squares, stop checking those and review only what falls below — that is
what the flag-threshold slider is for.

## Measured behaviour

### On a real chess book

*The Woodpecker Method*, 7 sheets imposed six-up, 222 diagrams set in Chess
Merida at about 23 pixels a square — small, dense, and printed with hatched dark
squares rather than a flat tint.

| | |
|---|---|
| diagrams found | **222 of 222**, no false positives |
| squares read correctly | **14,208 of 14,208 — 100.00%** |
| diagrams read perfectly | **222 of 222** |
| corrections needed | **none** |

That book is typeset in a chess font, so its text layer records the exact
position of every diagram and exactly where each one sits on the page. Which
means this is measured against real ground truth from a real book, not against
anything the tool drew itself. `tools/` has the harness.

Two caveats worth stating. Merida is one of the styles the packaged model is
trained on, so this is a favourable book — a title set in a font unlike anything
in training will be harder, which is what the held-out numbers below are for.
And getting here needed three fixes that only a real book could have prompted:
hatched squares and scanner noise, both below, and captions — this book prints
each game's players *above* its board, and reading only the band underneath had
been attaching every diagram to its neighbour's header, silently and
consistently.

### When the book is not a clean PDF

A clean PDF scoring perfectly says little on its own — plenty of chess books
only exist as photocopies, and plenty of readers photograph a page rather than
scan it. The same 222 diagrams, put through what actually happens to a page
(`tools/eval_stress.py`):

| condition | detection recall | squares | diagrams perfect |
|---|---|---|---|
| clean, 200 dpi | 100.0% | 100.00% | 222/222 |
| 150 dpi | 99.1% | 99.44% | 218/220 |
| 120 dpi | 99.5% | 99.48% | 214/221 |
| 100 dpi | 82.9% | 97.46% | 167/184 |
| photocopy — 150 dpi, soft, noisy, JPEG 70, 0.4° skew | 94.6% | 99.11% | 206/210 |
| poor scan — 120 dpi, blurred, noisy, JPEG 45, 0.8° skew | 65.3% | 88.04% | 7/145 |
| phone photo — 110 dpi, soft, noisy, JPEG 35, 1.5° skew | 69.8% | 42.71% | 0/155 |

**Ingest at 200 dpi if you can and 150 if you must.** Below about 120 dpi a
diagram this size is under fifteen pixels a square and detection starts losing
boards — not the classifier's fault, there is simply not enough board left. A
photocopy at a decent resolution is fine; a phone snapshot of a page is not, and
no amount of model is going to rescue it.

Those last two rows read 48% and 26% before the scanner-noise fix below — worse
than answering "empty" for every square — and 82%/55% after it. Training on the
resolutions and damage real books actually arrive in took the poor scan the rest
of the way to 88%.

### Finding the diagrams (generated books)

| | |
|---|---|
| 12 generated books including hatched and stippled boards | 94.8% recall at 100% precision |
| the same books with flat-tinted boards only | 97.2% recall at 100% precision |
| across 11 figurine styles, same diagram styles | 87.5%–95.8% recall, no false positives |
| lattice accuracy | within 2% of a cell on 99% of renders |
| the same books re-scanned: no text layer, skewed, JPEG at quality 72 | 32/32 diagrams, no false positives |

Detection is the settled part. It is not learned, it survives a bad scan, and it
genuinely does not care about the figurine style — swapping merida for alpha for
`letter` moves recall by a few points and misses the *same* diagrams, because
what defeats it is a board printed with no shading, no rules and no frame, not
the pieces standing on it.

Its threshold is set from the measured gap rather than by taste: over those 144
diagrams every false positive scored 0.150 or under and no true diagram scored
between 0.150 and 0.228, so the bar sits at 0.20, in the middle of the empty
band.

### Reading the pieces

This is the part that is hard, and the number that matters is accuracy on a
book set in a figurine font the model has **never seen** — because that is what
your book is. `alpha`, `leipzig`, `companion` and `chess7` are the classic
printed-book fonts and are deliberately kept out of training, so they are an
honest test rather than a rehearsal.

The packaged model is trained on 19 figurine styles — 16 real ones under
licences that allow redistribution, plus the three built in. Held-out results:

| | trained on 3 styles | **trained on 19 (shipped)** |
|---|---|---|
| synthetic validation, styles it *was* trained on | 99.98% | 99.58% |
| unseen book fonts, per square | 92.55% | **98.98%** |
| unseen book fonts, diagrams read perfectly | 16 of 45 | **31 of 45** |

Training on real figurine artwork rather than three lookalikes cut the errors on
an unseen book font from about five a diagram to about two thirds of one. That
is the single biggest accuracy change in the project, and it costs nothing at
run time — which is why `dgc pieces --fetch` is worth the two minutes.

The review loop then takes it the rest of the way:

| verified diagrams | `companion` | `leipzig` |
|---|---|---|
| 0 | 0.45 errors/diagram | 2.86 |
| 2 | 0.15 | 2.90 |
| 4 | 0.11 | 1.56 |
| 6 | 0.06 | 0.94 |

Neither of those is the model getting better; it is the exemplar bank filling up
with crops from your own book. On a *weak* model — one trained on a single style,
which is what a genuinely unfamiliar book looks like — the same loop goes from
24.0 errors a diagram to 4.5.

### Four measurements that changed the design

**Scanner noise read as pieces, found by stressing that same book.** Each square
crop used to be divided by its own standard deviation, on the reasoning that how
darkly a book prints says nothing about which piece is on the square. True — but
*how much contrast a crop has* says a great deal, and dividing it out threw that
away. An empty square has almost no variance, so on a noisy page the division
amplified sensor noise until it filled the range and the classifier saw a
high-contrast blob. Measured on the book, empty crops had a standard deviation
of 14.7 and occupied ones 24.9; standardising sent both to exactly 1.0. On one
page of a poor scan, 382 errors were the same mistake — an empty square called a
piece — and the whole scan scored below what answering "empty" everywhere would
have got. Flooring the divisor took a poor scan from 48% of squares to 81% and a
phone photo from 26% to 60%, with clean pages unchanged at 100%.



**Hatched squares, found by the first real book.** Presses cannot print grey, so
they print sparse black marks that average to grey — and a great many chess
books shade their dark squares that way. Measuring ink per cell straight, every
one of those squares reads as full, a board comes out with sixty of its
sixty-four cells "occupied", and the diagram is discarded for not being a chess
position. It was resolution-dependent too, which is worse: a coarse render blurs
the screen into a flat tint and finds the board, a finer one resolves the strokes
and does not. On the real book this threw away **65 of 222 diagrams**. A 3×3
median filter before thresholding erases marks a pixel or two wide and leaves a
piece's solid body untouched: **222 of 222**. The synthesiser now draws hatched
and stippled boards too — calibrated against that book's actual screen, roughly a
fifth coverage in near-black ink — so the case is in the tests from now on.



**The exemplar bank's weighting was fitted to the wrong benchmark — twice.** It
first spoke only where the model was unsure, which was tuned against books in
the styles the model already knew, where the model was right about everything
and the only measurable effect was the harm the bank could do. Re-measured on
unseen fonts, where the model is *confidently* wrong, that gate silenced the
bank exactly where it was needed. So it was changed to weight by label coverage
— and then the model got better, and coverage became the harmful one:

| weighting | weak model | shipped model |
|---|---|---|
| model alone | 11.34 | 0.94 |
| by the model's doubt | 10.91 | **0.65** |
| by label coverage | **6.43** | 2.63 |
| coverage × how lost the model looks *(shipped)* | 6.71 | 1.20 |

Nothing wins both columns, so the rule is chosen on the worst case rather than
the average: the hard column is the entire reason the bank exists, and being
second-best in the easy column costs half a correction per diagram on diagrams
that barely need reviewing. "How lost the model looks" is how much the two
readers disagree across the whole board — no labels needed, just the two
readings already in hand.

A bank built from a *single* verified diagram is also ignored outright. It has
one crop per class, no sense of how much a piece varies within a book, and just
enough coverage to sound confident; measured, it made readings worse than the
model alone. From the second diagram on it helps at every bank size.

**A renderer bug was masquerading as a model failure.** Several figurine sets,
merida among them, paint the white pieces' bodies with an SVG gradient, and
PyMuPDF silently drops gradient fills. Every white piece came through as a bare
dark outline, so the classifier was trained and tested on artwork where the two
colours were nearly the same — and every single error it made on merida was a
colour flip, with the piece type always right. Rendering through Cairo instead
took merida from 87% to 100%. `dgc pieces` now refuses any style whose white and
black artwork do not come out distinguishable, so the next such failure is loud
rather than silent.

## Commands

| | |
|---|---|
| `dgc ingest FILE.pdf` | find and read the diagrams; `--pages 10-40`, `--dpi` |
| `dgc review` | the review server; `--port`, `--model` |
| `dgc train` | train on synthetic diagrams plus everything you have verified |
| `dgc reread` | re-read stored diagrams with the current model |
| `dgc accuracy` | how often the model agrees with your corrections |
| `dgc export --format board\|fen\|csv\|pgn\|json` | the positions found so far |
| `dgc status` | what is in the workspace |
| `dgc models --activate N` | switch between trained models |
| `dgc pieces --fetch` | download figurine styles; `--shippable-only` filters by licence |
| `dgc demo-book OUT.pdf` | generate a sample book; `--style` pins the figurine font |

Everything lives in a workspace directory (`.diagramchess` by default, or
`-w somewhere-else`): the SQLite database, the cached crops, and the model
checkpoints. Deleting it loses your corrections; nothing else in the tool holds
state.

Re-ingesting a book you have already ingested is safe. Diagrams are recognised
by where they sit on the page, so a second pass at a different resolution keeps
everything you verified.

## When it gets one wrong

- **A diagram was missed.** Detection refuses rather than guesses when a board
  is faint. Lower the bar with `SCORE_THRESHOLD` in `detect.py`, or re-ingest
  at a higher `--dpi`.
- **Every square is wrong.** The lattice is cut in the wrong place. Check the
  crop in the review screen — the cell boundaries should fall between the
  pieces.
- **A few squares are wrong.** Correct them and press `a` on the next diagram;
  with exemplars from the same book, the same piece usually stops being a
  problem immediately.
- **The board is upside down.** Press `f`. Orientation is guessed from where
  the pieces sit, which endgames can defeat.
- **The wrong side to move.** Press `t`. It is read from the caption printed
  around the diagram when the book prints one — a scanned book has no text
  layer, so there every diagram starts at "white to move" until you say
  otherwise. (An OCR pass over the caption band would fix that; it is not
  built.)

## The Lichess links

Positions are handed over as URLs in Lichess's path form — the FEN with its
spaces written as underscores:

```
https://lichess.org/analysis/standard/4k3/r1q3p1/.../RNB1KR2_w_Q_-_0_1
https://lichess.org/editor/4k3/r1q3p1/.../RNB1KR2_w_Q_-_0_1
```

The editor link is the better landing place when a position still needs a
tweak. Every screen also offers **Copy FEN**, which works regardless of what
any site does with its URLs.

A diagram cannot say whose move it is, whether anyone has castled, or which way
round the board is meant to be read. The side to move is taken from the caption
printed around the diagram when there is one; castling rights are granted when
king and rook are both still on their home squares, the same rule board editors
use; orientation is guessed from where the pieces sit. All three are one
keystroke to change in review.

## Piece artwork

Three styles ship with the tool: the vector set from python-chess, and the chess
glyphs in DejaVu Sans and FreeSerif. Three is not many, and **more styles in
training is the cheapest accuracy you can buy** — so:

```
dgc pieces --fetch     # ~40 more styles from Lichess, a few megabytes
dgc pieces             # what you have, and what each one's licence allows
dgc train              # picks them up automatically
```

The licences differ per style — GPL, MIT, Apache, CC0 and CC BY are fine to
redistribute a trained model under; several are non-commercial and one forbids
derivatives — so nothing is downloaded unless you ask and nothing is vendored
into this repository. `dgc pieces` prints the licence next to each style and
marks which ones a shipped model may be trained on.

You can also add your own: a directory per style, twelve files named `wK`,
`bQ` and so on, as SVG or PNG, under `.diagramchess/pieces/`. A style is
rejected if it is incomplete, or if its white and black artwork do not render
distinguishably — see the renderer bug above for why that check is there.

## Licence

GPL-3.0-or-later, because it links python-chess and uses its piece artwork.
