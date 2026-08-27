import pytest

from diagramchess.board import (
    BLACK_AT_BOTTOM, WHITE_AT_BOTTOM, BoardError, BoardMatrix,
    guess_orientation, guess_side_to_move,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_fen_round_trip():
    assert BoardMatrix.from_fen(START).to_fen() == START


def test_rows_are_display_order_with_white_at_the_bottom():
    board = BoardMatrix.from_fen(START)
    assert board.rows[0] == list("rnbqkbnr")   # top of the picture is rank 8
    assert board.rows[7] == list("RNBQKBNR")
    assert board.square_name(0, 0) == "a8"
    assert board.square_name(7, 7) == "h1"


def test_flipping_reads_the_same_picture_from_the_other_side():
    board = BoardMatrix.from_fen(START)
    flipped = board.flipped()
    assert flipped.orientation == BLACK_AT_BOTTOM
    assert flipped.square_name(0, 0) == "h1"
    # Same pixels, mirrored reading: the back ranks swap colour.
    assert flipped.placement().startswith("RNBKQBNR")


def test_castling_is_inferred_from_home_squares():
    board = BoardMatrix.from_fen(START)
    assert board.infer_castling() == "KQkq"
    board[7, 7] = "."          # remove the h1 rook
    assert board.infer_castling() == "Qkq"
    board[7, 4] = "."          # remove the white king
    assert board.infer_castling() == "kq"


def test_castling_can_be_stated_explicitly():
    board = BoardMatrix.from_fen(START)
    board.castling = "-"
    assert board.to_fen().split()[2] == "-"


def test_lichess_urls():
    board = BoardMatrix.from_fen(START)
    assert board.lichess_url().startswith("https://lichess.org/analysis/standard/rnbqkbnr/")
    assert "_w_KQkq_" in board.lichess_url()
    assert board.lichess_url("editor").startswith("https://lichess.org/editor/")
    with pytest.raises(ValueError):
        board.lichess_url("nowhere")


def test_orientation_guess_follows_the_pieces():
    normal = BoardMatrix.from_fen(START)
    assert guess_orientation(normal.rows) == WHITE_AT_BOTTOM
    upside_down = [list(reversed(r)) for r in reversed(normal.rows)]
    assert guess_orientation(upside_down) == BLACK_AT_BOTTOM


def test_orientation_guess_defaults_when_one_side_is_absent():
    board = BoardMatrix.empty()
    board[4, 4] = "K"
    assert guess_orientation(board.rows) == WHITE_AT_BOTTOM


@pytest.mark.parametrize("caption,expected", [
    ("Diagram 4: Black to play", "b"),
    ("White to move and win", "w"),
    ("Black to move.", "b"),
    ("W: Kf1", "w"),
    ("A quiet position", None),
    ("White to play, or is it Black to play?", None),
    ("", None),
    (None, None),
])
def test_side_to_move_is_read_from_the_caption(caption, expected):
    assert guess_side_to_move(caption) == expected


def test_problems_flags_impossible_positions():
    board = BoardMatrix.empty()
    problems = board.problems()
    assert any("white king" in p for p in problems)
    assert any("black king" in p for p in problems)

    board = BoardMatrix.from_fen(START)
    assert board.problems() == []

    board[0, 0] = "P"          # a white pawn on a8
    assert any("rank 8" in p for p in board.problems())


def test_malformed_boards_are_rejected():
    with pytest.raises(BoardError):
        BoardMatrix([["."] * 8] * 7)
    with pytest.raises(BoardError):
        BoardMatrix.from_labels("." * 63)
    with pytest.raises(BoardError):
        BoardMatrix([["Z"] + ["."] * 7] + [["."] * 8] * 7)
    with pytest.raises(BoardError):
        BoardMatrix.from_fen(START, orientation="sideways")


def test_exported_fens_parse_as_chess_positions():
    """python-chess is the stand-in for whatever reads the FEN at the other end."""
    import chess

    for fen in (
        START,
        "8/8/4k3/8/2K5/8/6R1/8 w - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "2kr3r/ppp2ppp/2n1b3/3q4/3P4/2N1BN2/PPP2PPP/R2Q1RK1 w - - 0 1",
    ):
        board = BoardMatrix.from_fen(fen)
        parsed = chess.Board(board.to_fen())
        assert parsed.board_fen() == board.placement()
        assert parsed.turn == (board.side_to_move == "w")


def test_a_flipped_reading_still_produces_a_parseable_fen():
    import chess

    board = BoardMatrix.from_fen(START).flipped()
    chess.Board(board.to_fen())     # raises if the placement is malformed


def test_a_puzzle_fragment_still_exports_even_though_it_is_not_a_legal_game():
    """Books print positions with no kings; we warn but must not refuse to export."""
    board = BoardMatrix.empty()
    board[3, 3] = "Q"
    board[4, 4] = "r"
    fen = board.to_fen()
    assert fen.startswith("8/8/8/3Q4/4r3/8/8/8")
    assert board.problems()                 # warned about
    assert board.lichess_url()              # but still exported
