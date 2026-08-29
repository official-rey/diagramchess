# Getting started

A step-by-step guide for someone who has not used a terminal much. If you are
comfortable with Python and virtual environments, the README's *Installing*
section says the same thing in six lines.

Everything below has been run start to finish on a clean machine.

---

## What you will end up with

A command called `dgc`. You point it at a chess book PDF, it finds every
diagram, reads the position off each one, and gives you a Lichess link for each.
There is a review screen for checking its work, and what you correct there makes
it better at the rest of your book.

Nothing is uploaded anywhere. It all runs on your computer.

---

## Step 1 — Open a terminal

This is the window you type commands into.

- **Windows** — press `Win`, type `powershell`, press Enter.
- **macOS** — press `Cmd + Space`, type `terminal`, press Enter.
- **Linux** — `Ctrl + Alt + T`, or search your applications for "Terminal".

You will see a line ending in `>` or `$`. That is the prompt; you type after it
and press Enter. Leave the window open for the rest of this guide.

---

## Step 2 — Check you have Python

Type this and press Enter:

```
python3 --version
```

**On Windows, type `python --version` instead** — Windows spells it without the 3.

You want to see `Python 3.10` or higher:

```
Python 3.11.9
```

**If it says that, skip to Step 3.**

**If you get an error, or a version below 3.10**, install Python:

1. Go to <https://www.python.org/downloads/>
2. Click the big yellow download button, run the installer.
3. **On Windows this matters:** tick **"Add python.exe to PATH"** on the first
   screen of the installer before clicking Install. It is easy to miss and
   nothing works without it.
4. Close the terminal, open a new one, and try `python3 --version` again.

---

## Step 3 — Install the tool

Two commands. The first makes a private folder for the tool so it cannot
disturb anything else on your computer; the second downloads it.

**macOS / Linux:**

```bash
python3 -m venv ~/chess-tool
~/chess-tool/bin/pip install "diagramchess[ml] @ git+https://github.com/official-rey/diagramchess@claude/chess-pdf-diagram-extractor-plh834"
```

**Windows (PowerShell):**

```powershell
python -m venv $HOME\chess-tool
$HOME\chess-tool\Scripts\pip install "diagramchess[ml] @ git+https://github.com/official-rey/diagramchess@claude/chess-pdf-diagram-extractor-plh834"
```

The second command takes a few minutes and prints a great deal. Most of it is
PyTorch, which is large. As long as the last lines do not say `ERROR`, it worked.

Check it:

```bash
~/chess-tool/bin/dgc --version          # Windows: $HOME\chess-tool\Scripts\dgc --version
```

```
diagramchess 0.1.0
```

> **Why the long path?** So you never have to remember to "activate" anything.
> `~/chess-tool/bin/dgc` always works, from any folder. If you would rather just
> type `dgc`, see *Shortening the command* at the end.

> **`git` not found?** The install line needs Git. Install it from
> <https://git-scm.com/downloads>, then run the command again.

---

## Step 4 — Try it on a sample book

Before using a real book, make one. The tool can generate a sample chess book
with diagrams it knows the answers to.

```bash
mkdir ~/chess
cd ~/chess
~/chess-tool/bin/dgc demo-book sample.pdf
```

```
wrote sample.pdf with 12 diagrams (ground truth in sample.truth.json)
```

`cd ~/chess` moved you into that folder — commands from now on run there. Now
read the sample:

```bash
~/chess-tool/bin/dgc ingest sample.pdf
```

```
sample.pdf: book 1: 8 pages, 12 diagrams detected, 12 new, 0 already known, 12 read
```

And see what it found:

```bash
~/chess-tool/bin/dgc export --status pending --format board
```

You should get chess boards drawn in the terminal, each with a Lichess link
underneath. Click one — it opens that exact position on Lichess.

**If that worked, the tool is installed correctly.** On to your own book.

---

## Step 5 — Use your own book

Put your PDF somewhere you can find it. The simplest thing is to copy it into
the `~/chess` folder you just made, then:

```bash
~/chess-tool/bin/dgc ingest my-book.pdf
```

Replace `my-book.pdf` with your file's actual name. If the name has spaces in
it, wrap it in quotes: `"My Chess Book.pdf"`.

A few hundred pages takes a couple of minutes. You will see:

```
my-book.pdf: book 1: 224 pages, 187 diagrams detected, 187 new, 0 already known, 187 read
```

> **Trying a big book?** Add `--pages 40-60` the first time to do just a few
> pages and see how it goes before committing to the whole thing.

---

## Step 6 — Check its work

```bash
~/chess-tool/bin/dgc review
```

```
review UI on http://127.0.0.1:8765
```

Open <http://localhost:8765> in your browser. Click a book, then a diagram.

You will see the diagram as printed on the left, and a chess board in the middle
showing what the tool read. The board is not a picture of your book — it is the
tool's answer, with the printed page faintly underneath it. Where the two agree
the pieces line up cleanly; where they disagree you see two pieces at once, and
that is the square to fix.

Squares the tool was unsure of are ringed in orange or red and show how sure it
was. Everything unmarked, it is confident about.

The list is ordered with the diagrams it is *least sure about* first, so if the
first few are right, the rest very likely are too.

To correct a square: click it, then press a letter.

| key | does |
|---|---|
| `k` `q` `r` `b` `n` `p` | put a **black** king, queen, rook, bishop, knight, pawn |
| `Shift` + those | put a **white** one |
| `space` | empty the square |
| arrow keys | move to another square without the mouse |
| `Tab` | jump to the next square it is unsure about |
| `f` | flip the board round |
| `t` | switch whose move it is |
| `c` | cycle: reading over the picture / reading only / picture only |
| `Enter` | save and go to the next diagram |
| `l` | open this position on Lichess |
| `Esc` | back to the list |

`Tab` is the one to lean on. It takes you straight to the squares the tool
doubted, so a diagram is usually a couple of `Tab` presses and an `Enter`.

Saving records all 64 squares, including the ones it got right — those teach it
that its confident answers are correct.

Press `Ctrl + C` in the terminal to stop the review server when you are done.

---

## Step 7 — Get the positions out

```bash
# boards and Lichess links, in the terminal
~/chess-tool/bin/dgc export --format board

# a PGN file you can open in any chess program
~/chess-tool/bin/dgc export --format pgn > my-book.pgn

# a spreadsheet
~/chess-tool/bin/dgc export --format csv > my-book.csv
```

By default these give you the diagrams you have **verified**. Add
`--status all` for everything, verified or not.

---

## If it is getting things wrong

First, ask it how it is doing:

```bash
~/chess-tool/bin/dgc accuracy
```

It compares its own readings against your corrections and tells you what kind of
mistake it is making. Then teach it your book:

```bash
~/chess-tool/bin/dgc pieces --fetch     # downloads ~40 chess piece designs
~/chess-tool/bin/dgc train              # 20-45 minutes, leave it running
~/chess-tool/bin/dgc reread             # re-read the book with what it learned
```

`train` uses both those piece designs and every square you corrected in the
review screen. Twenty checked diagrams is enough to make a real difference.

Quick fixes for specific problems:

| what you see | what to do |
|---|---|
| A whole diagram is nonsense | The grid was cut in the wrong place — check in the review screen that the square edges fall *between* the pieces |
| The board is upside down | Press `f` |
| Wrong side to move | Press `t` |
| It missed diagrams entirely | Re-run with `--dpi 300` — the diagrams may be small |
| Something that is not a chess diagram | Click **Not a diagram** |

---

## Things worth knowing

**Where your work is kept.** A folder called `.diagramchess` inside whatever
folder you ran the commands from. It holds everything — the diagrams, your
corrections, the trained models. Back it up if you care about it; delete it to
start completely fresh. (The dot at the front means your file browser hides it
by default.)

**One folder per book, or one for all of them?** Either. Running `dgc` from the
same folder adds books to the same collection. To keep a book separate, make a
new folder and run from there.

**Re-running `ingest` is safe.** It recognises diagrams it has already seen and
keeps everything you verified.

---

## Shortening the command

If typing `~/chess-tool/bin/dgc` gets tiresome, you can "activate" the tool for
a terminal session — after which plain `dgc` works, until you close the window:

```bash
source ~/chess-tool/bin/activate        # macOS / Linux
$HOME\chess-tool\Scripts\Activate.ps1   # Windows PowerShell
```

```bash
dgc status
```

You will see `(chess-tool)` at the start of your prompt while it is active.

---

## When something goes wrong

**`dgc: command not found`** — you either mistyped the path or the install did
not finish. Run the check from Step 3 again.

**`python3: command not found` on Windows** — use `python`, not `python3`.

**Something about `cairo`** — run `~/chess-tool/bin/dgc pieces`; if it mentions
Cairo, follow what it says. On Ubuntu that is `sudo apt install libcairo2`, on
macOS `brew install cairo`. This does not affect reading books; it only matters
if you go on to `dgc train`.

**"0 diagrams detected"** — the book may be a low-quality scan. Try
`--dpi 300`. If the diagrams are photographs of a board rather than printed
diagrams, this tool will not read them.

**Readings are poor on your book** — likely a chess font the tool has not seen.
Verify ten or twenty diagrams in the review screen, then `dgc train` and
`dgc reread`. That is what the review screen is for.
