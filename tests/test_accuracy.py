from diagramchess.accuracy import measure
from diagramchess.board import BoardMatrix

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _diagram(workspace, book, index, predicted, actual, confidence=0.9):
    diagram = workspace.add_diagram(
        book, 0, (float(index), 0.0, 100.0, 100.0), {"x0": 0}, "contour", 0.9, f"c{index}.png")
    workspace.set_prediction(diagram, predicted, [confidence] * 64, "f", "white", "w", None)
    workspace.save_review(diagram, actual, "white", "w", "f")
    return diagram


def test_nothing_verified_measures_nothing(workspace):
    report = measure(workspace)
    assert report.diagrams == 0
    assert "nothing verified yet" in report.describe()


def test_a_perfect_reading_scores_perfectly(workspace):
    book = workspace.add_book("/a.pdf", "d", 1, 200)
    labels = BoardMatrix.from_fen(START).flat()
    _diagram(workspace, book, 0, labels, labels)
    report = measure(workspace)
    assert report.diagrams == 1
    assert report.square_accuracy == 1.0
    assert report.diagram_accuracy == 1.0
    assert report.corrections_per_diagram == 0.0


def test_mistakes_are_counted_and_named(workspace):
    book = workspace.add_book("/a.pdf", "d", 1, 200)
    actual = BoardMatrix.from_fen(START).flat()
    predicted = list(actual)
    predicted[0] = "n"          # a rook read as a knight
    predicted[63] = "Q"         # a rook read as a queen
    _diagram(workspace, book, 0, predicted, actual)

    report = measure(workspace)
    assert report.correct_squares == 62
    assert report.diagram_accuracy == 0.0
    assert report.corrections_per_diagram == 2.0
    worst = dict(((a, p), n) for a, p, n in report.worst_confusions())
    assert worst[("r", "n")] == 1
    assert worst[("R", "Q")] == 1


def test_confidence_bands_show_what_a_threshold_would_cost(workspace):
    book = workspace.add_book("/a.pdf", "d", 1, 200)
    actual = BoardMatrix.from_fen(START).flat()
    _diagram(workspace, book, 0, list(actual), actual, confidence=0.995)
    wrong = list(actual)
    wrong[0] = "n"
    _diagram(workspace, book, 1, wrong, actual, confidence=0.80)

    report = measure(workspace)
    above, missed = report.missed_at_confidence(0.99)
    assert above == 64 and missed == 0        # the confident diagram, all correct
    above, missed = report.missed_at_confidence(0.5)
    assert above == 128 and missed == 1


def test_per_book_totals(workspace):
    first = workspace.add_book("/a.pdf", "d1", 1, 200)
    second = workspace.add_book("/b.pdf", "d2", 1, 200)
    labels = BoardMatrix.from_fen(START).flat()
    _diagram(workspace, first, 0, labels, labels)
    broken = list(labels)
    broken[5] = "."
    _diagram(workspace, second, 1, broken, labels)
    report = measure(workspace)
    assert report.per_book[first] == (1, 1)
    assert report.per_book[second] == (0, 1)
    assert measure(workspace, book_id=first).diagrams == 1


def test_advice_changes_with_the_evidence(workspace):
    book = workspace.add_book("/a.pdf", "d", 1, 200)
    labels = BoardMatrix.from_fen(START).flat()
    _diagram(workspace, book, 0, labels, labels)
    assert "verify a few more" in measure(workspace).advice()

    for index in range(1, 10):
        _diagram(workspace, book, index, labels, labels)
    assert "spot-checking" in measure(workspace).advice()
