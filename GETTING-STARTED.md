# Getting started

A step-by-step guide for someone who has not used a terminal much. If you are
comfortable with Python and virtual environments, the README's *Installing*
section says the same thing in six lines.

Everything below has been run start to finish on a clean machine.

---

## What you will end up with

A window on your computer that you drop a chess book PDF into. It finds every
diagram, reads the position off each one, and gives you a Lichess link for each.
There is a review screen for checking its work, and what you correct there makes
it better at the rest of your book.

You need a terminal twice: once to install it, once to start it. After Step 8
even that goes away and it becomes an icon you double-click.

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

## Step 4 — Start it

```bash
~/chess-tool/bin/dgc app          # Windows: $HOME\chess-tool\Scripts\dgc app
```

A browser window opens on the tool itself. Everything from here happens in that
window — you will not need to type another command.

```
diagramchess on http://127.0.0.1:8765
workspace: /home/you/.diagramchess
press Ctrl-C here to stop the tool
```

Leave the terminal window alone while you work; it is the tool running. Closing
it stops the tool. (Step 8 replaces even this with a desktop shortcut.)

---

## Step 5 — Open a book

In the window, click **Try a sample book** first. It draws a small chess book,
reads it, and drops you straight into the review screen — that is the whole
tool working end to end, and it takes about a minute.

Then do it with yours: go back with **‹ books**, and either drag your PDF onto
the dashed panel or click **Choose a PDF…**. It reads the book, showing you
which page it is on and how many diagrams it has found, and takes you to the
review screen when it finishes.

Nothing is uploaded anywhere. The file is copied into the tool's own folder on
your computer and read there.

> **Long book?** Click **Options** and put something like `40-60` in *pages* to
> try a stretch of it first. **Missed diagrams?** Options → detail →
> *finest*, which helps on small or faintly printed diagrams.

---

## Step 6 — Check its work

You are already there after opening a book; **Review** next to a book in the
list gets you back to it.

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

Every diagram has an **Open on Lichess** button, which is usually all you want.
To take the whole book somewhere else, open a *second* terminal window (the
first one is busy running the tool) and:

```bash
cd ~            # or wherever you started the tool from
# boards and Lichess links, printed out
~/chess-tool/bin/dgc export --format board

# a PGN file you can open in any chess program
~/chess-tool/bin/dgc export --format pgn > my-book.pgn

# a spreadsheet
~/chess-tool/bin/dgc export --format csv > my-book.csv
```

By default these give you the diagrams you have **verified**. Add
`--status all` for everything, verified or not.

---

## Step 8 — Never open a terminal again

Once, run:

```bash
~/chess-tool/bin/dgc install-launcher
```

It puts a shortcut where your computer keeps its programs — the applications
menu on Linux, the Applications folder on macOS, the Desktop on Windows — and
prints where it went. From then on, double-clicking it starts the tool and
opens the window, with no terminal at all.

The shortcut remembers which folder your books live in, so run this from the
folder you have been working in. `dgc install-launcher --remove` takes it away
again.

---

## If it is getting things wrong

First, ask it how it is doing:

```bash
~/chess-tool/bin/dgc accuracy
```

It compares its own readings against your corrections and tells you what kind of
mistake it is making.

Teaching it your book is done from the window, at the bottom of the home page:

1. **Download figurine styles** — about forty chess piece designs to train
   against. Worth doing once.
2. **Train on my corrections** — 20 to 45 minutes. Leave the window open; the
   progress bar shows which round it is on and how well it is doing.
3. **Read every diagram again** — re-reads the book with what it just learned.

Training uses both those piece designs and every square you corrected in the
review screen. Twenty checked diagrams is enough to make a real difference.

Quick fixes for specific problems:

| what you see | what to do |
|---|---|
| A whole diagram is nonsense | The grid was cut in the wrong place — check in the review screen that the square edges fall *between* the pieces |
| The board is upside down | Press `f` |
| Wrong side to move | Press `t` |
| It missed diagrams entirely | Open the book again with Options → detail → *finest* |
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

If you would rather stay in the terminal but not type the long path, you can
"activate" the tool for a session — after which plain `dgc` works until you
close the window:

```bash
source ~/chess-tool/bin/activate        # macOS / Linux
$HOME\chess-tool\Scripts\Activate.ps1   # Windows PowerShell
```

```bash
dgc app
```

You will see `(chess-tool)` at the start of your prompt while it is active.

---

## When something goes wrong

**`dgc: command not found`** — you either mistyped the path or the install did
not finish. Run the check from Step 3 again.

**`python3: command not found` on Windows** — use `python`, not `python3`.

**The window did not open** — go to <http://localhost:8765> yourself. If the
terminal said a port was busy, it will have printed the number it used instead.

**Something about `cairo`** — run `~/chess-tool/bin/dgc pieces`; if it mentions
Cairo, follow what it says. On Ubuntu that is `sudo apt install libcairo2`, on
macOS `brew install cairo`. This does not affect reading books; it only matters
if you go on to `dgc train`.

**"No diagrams found in that book"** — it may be a low-quality scan. Open it
again with Options → detail → *finest*. If the diagrams are photographs of a
real board rather than printed diagrams, this tool will not read them.

**Readings are poor on your book** — likely a chess font the tool has not seen.
Verify ten or twenty diagrams in the review screen, then `dgc train` and
`dgc reread`. That is what the review screen is for.
