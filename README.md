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
dgc demo-book sample.pdf      # a generated chess book, if you want to try it first
dgc train                     # a cold-start classifier; the default run took 27 min on 4 CPU cores
dgc ingest your-book.pdf      # find the diagrams and read them
dgc review                    # check them; corrections are saved as training data
dgc accuracy                  # is it good enough to stop checking every one?
dgc train && dgc reread       # fold your corrections back in
```

## Installing

```
python -m pip install -e '.[ml]'
```

The `ml` extra pulls in PyTorch, which is only needed for training and for the
neural classifier. Everything else — finding diagrams, the review screen,
exporting FENs, and reading a book from exemplars you have verified — works
without it.

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
classified 13 ways (12 pieces and empty) by a small convolutional net. Crops
are standardised individually, so how darkly a book prints carries no
information the net can lean on.

**Learning your book.** Alongside the net, an **exemplar bank** holds the crops
you have verified in the book you are reading, matched by nearest neighbour.
The net knows what chess pieces look like in general; the bank knows what they
look like *in this book*. The bank's weight grows as it fills, and the net stays
in the mix so a piece the bank has never seen is still reachable.

## The review screen

`dgc review` serves it on `localhost:8765`. The queue is ordered by how unsure
the model is, so the diagrams that teach it most come first. Within a diagram,
the cursor starts on the least confident square.

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
agreement looks clean and disagreement looks doubled. Squares the model is
unsure about are outlined; the threshold is a slider.

Saving records all 64 squares, including the ones the model got right — those
are what teach it that its confident answers are correct.

## Knowing when it is good enough

`dgc accuracy` compares what the model said against what you said, on the
diagrams you have already checked. That is the only number that answers the
question you actually care about, and it is the one to watch:

```
measured against 24 diagram(s) you verified (1536 squares)
  squares read correctly:   99.61%
  diagrams read perfectly:   87.50%
  corrections per diagram:    0.25
  above 99% confidence: 1450 squares, 0 of them wrong (0.00%)
  mistakes it makes most:
      white bishop read as white pawn: 3
  -> the model now reads whole diagrams correctly 88% of the time; you could
     switch to spot-checking the low-confidence ones only
```

The confidence bands are the useful part: if nothing above 99% confidence has
ever been wrong across a few hundred squares, you can stop checking those and
review only what falls below.

## Measured behaviour

Numbers below come from the tools in `tools/`, run against generated books that
include crosstable pages as distractors.

| | |
|---|---|
| diagram detection | 97% recall at 100% precision |
| lattice accuracy | within 2% of a cell on 99% of renders |
| classifier, synthetic validation | 99.9% of squares |
| exemplars alone, unseen position in a known style | 58–64 of 64 squares |

**Read the classifier number with suspicion.** It says the model reads diagrams
*we drew* almost perfectly, and the tool draws them from three piece sets that
resemble each other far more than a real book's figurine font resembles any of
them. Accuracy on your book will start lower. That gap is the reason the review
loop exists, and `dgc accuracy` is what measures it honestly.

## Commands

| | |
|---|---|
| `dgc ingest FILE.pdf` | find and read the diagrams; `--pages 10-40`, `--dpi` |
| `dgc review` | the review server; `--port`, `--model` |
| `dgc train` | train on synthetic diagrams plus everything you have verified |
| `dgc reread` | re-read stored diagrams with the current model |
| `dgc accuracy` | how often the model agrees with your corrections |
| `dgc export --format fen\|csv\|pgn\|json` | the positions found so far |
| `dgc status` | what is in the workspace |
| `dgc models --activate N` | switch between trained models |
| `dgc demo-book OUT.pdf` | generate a sample book with known contents |

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
  around the diagram when the book prints one.

## Piece artwork

Synthetic training diagrams are drawn from the vector set that ships with
python-chess and from the chess glyphs in DejaVu Sans and FreeSerif. If your
books use a figurine style unlike any of those, drop PNG files named `wK.png`,
`bQ.png` and so on into a directory per style and point
`available_piece_sets(extra_dir=...)` at it — more styles in training is the
cheapest way to raise cold-start accuracy.

## Licence

GPL-3.0-or-later, because it links python-chess and uses its piece artwork.
