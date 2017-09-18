from .piece_type import PieceType


class Pawn(PieceType):
    """
    Pawn move logic.

    TODO: Restrict en passant captures to (a) just the move after opponent's
    two-square move, and (b) just after two-space moves, not also one + one
    square moves forward.
    """

    @classmethod
    def move_yx_yx(cls, board, from_yx, to_yx, promote_to, has_my_king_moved,
                   dry_run=False):
        if not 0 <= to_yx[0] < 8 or not 0 <= to_yx[1] < 8:
            return False
        y = from_yx[0]
        x = from_yx[1]
        if to_yx[0] == 7:
            assert promote_to is not None
        else:
            assert promote_to is None
        if to_yx[0] - y == 1 and to_yx[1] - x == 0 and \
                board[to_yx] == cls.space:
            # One step forward.
            if not dry_run:
                if to_yx[0] == 7:
                    board[to_yx] = promote_to
                else:
                    board[to_yx] = board[from_yx]
                board[from_yx] = cls.space
            return True
        elif to_yx[0] - y == 1 and to_yx[1] - x in {-1, 1} and \
                cls.int2whose[board[to_yx]] == cls.theirs:
            # Normal capture.
            if not dry_run:
                if to_yx[0] == 7:
                    board[to_yx] = promote_to
                else:
                    board[to_yx] = board[from_yx]
                board[from_yx] = cls.space
            return True
        elif y == 1 and to_yx[0] == 3 and board[2, x] == cls.space and \
                board[3, x] == cls.space:
            # Two squares forward from second rank.
            if not dry_run:
                board[to_yx] = board[from_yx]
                board[from_yx] = cls.space
            return True
        elif y == 4 and to_yx[0] - y == 1 and to_yx[1] - x in {-1, 1} and \
                board[to_yx] == cls.space and \
                board[y, to_yx[1]] == cls.their_pawn:
            # En passant capture from fifth rank.
            if not dry_run:
                board[to_yx] = board[from_yx]
                board[from_yx] = cls.space
                board[y, to_yx[1]] = cls.space
            return True
        else:
            return False

    @classmethod
    def each_possible_move(cls, board, from_yx, has_my_king_moved):
        y, x = from_yx
        promote_to = cls.my_queen if y + 1 == 7 else None

        # One square forward
        to_yx = y + 1, x
        if cls.move_yx_yx(board, from_yx, to_yx, promote_to, False, True):
            yield to_yx

        # Two squares forward.
        to_yx = y + 2, x
        if cls.move_yx_yx(board, from_yx, to_yx, promote_to, False, True):
            yield to_yx

        # One step diagonally (normal or en passant capture).
        to_yx = y + 1, x - 1
        if cls.move_yx_yx(board, from_yx, to_yx, promote_to, False, True):
            to_yx
        to_yx = y + 1, x + 1
        if cls.move_yx_yx(board, from_yx, to_yx, promote_to, False, True):
            to_yx

    @classmethod
    def find_origin_forward(cls, board, restrict_yx, to_yx):
        restrict_y, restrict_x = restrict_yx
        to_y, to_x = to_yx

        if restrict_x not in {None, to_x}:
            return []

        y = to_y - 1
        x = to_x
        if board[y, x] == cls.my_pawn and restrict_y in {None, y}:
            return [(y, x)]
        elif to_y == 3 and board[2, x] == cls.space and \
                board[1, x] == cls.my_pawn and restrict_y in {None, 1}:
            return [(1, x)]
        else:
            return []

    @classmethod
    def find_origin_en_passant(cls, board, restrict_yx, to_yx):
        restrict_y, restrict_x = restrict_yx
        to_y, to_x = to_yx

        if board[to_y - 1, to_x] != cls.their_pawn or \
                board[to_y, to_x] != cls.space:
            return []

        ret = []

        y = to_y - 1
        x = to_x - 1
        if 0 <= y < 8 and 0 <= x < 8 and board[y, x] == cls.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        y = to_y - 1
        x = to_x + 1
        if 0 <= y < 8 and 0 <= x < 8 and board[y, x] == cls.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        return ret

    @classmethod
    def find_origin_capture(cls, board, restrict_yx, to_yx):
        restrict_y, restrict_x = restrict_yx
        to_y, to_x = to_yx

        ret = []

        y = to_y - 1
        x = to_x - 1
        if 0 <= y < 8 and 0 <= x < 8 and board[y, x] == cls.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        y = to_y - 1
        x = to_x + 1
        if 0 <= y < 8 and 0 <= x < 8 and board[y, x] == cls.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        return ret

    @classmethod
    def find_origin(cls, board, restrict_yx, to_yx):
        restrict_y, restrict_x = restrict_yx
        to_y, to_x = to_yx
        ret = []
        if board[to_yx] == cls.space:
            ret += cls.find_origin_forward(board, restrict_yx, to_yx)
            ret += cls.find_origin_en_passant(board, restrict_yx, to_yx)
        else:
            ret += cls.find_origin_capture(board, restrict_yx, to_yx)
        return ret

    @classmethod
    def is_capture(cls, board, from_yx, to_yx):
        y, x = from_yx
        if y == 4 and to_yx[0] - y == 1 and to_yx[1] - x in {-1, 1} and \
                board[to_yx] == cls.space and \
                board[y, to_yx[1]] == cls.their_pawn:
            return True
        else:
            return cls.int2whose[board[to_yx]] == cls.theirs
