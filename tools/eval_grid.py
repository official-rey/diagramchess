"""Measure how accurately the grid fitter and detector land on rendered diagrams."""
import random, sys, statistics
sys.path.insert(0, "src")
import numpy as np
from diagramchess.board import BoardMatrix
from diagramchess.pieces import available_piece_sets
from diagramchess.render import random_style, render_diagram
from diagramchess.grid import fit_grid
from diagramchess.synth import random_position

def main(n=120, seed=0):
    rng = random.Random(seed)
    sets = available_piece_sets()
    errs, step_errs, bad = [], [], 0
    for i in range(n):
        board = random_position(rng)
        style = random_style(rng, rng.choice(sets))
        style.cell_px = rng.randint(24, 60)
        r = render_diagram(board, style)
        g = fit_grid(r.image)
        dx = abs(g.x0 - r.grid.x0) / r.grid.step_x
        dy = abs(g.y0 - r.grid.y0) / r.grid.step_y
        ds = abs(g.step_x - r.grid.step_x) / r.grid.step_x
        errs.append(max(dx, dy)); step_errs.append(ds)
        if max(dx, dy) > 0.12 or ds > 0.03:
            bad += 1
            if bad <= 6:
                print(f"  miss #{i}: set={style.piece_set.name} cell={style.cell_px} border={style.border_width} "
                      f"coords={style.coordinates} lines={style.grid_line} off={max(dx,dy):.3f} step={ds:.3f} L={g.line_score:.2f} C={g.checker_score:.2f}")
    print(f"n={n} median_offset={statistics.median(errs):.4f} cells  p90={np.percentile(errs,90):.4f}  "
          f"median_step_err={statistics.median(step_errs):.4f}  bad={bad}/{n}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
