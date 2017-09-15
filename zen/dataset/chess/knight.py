from .piece_type import PieceType


class Knight(PieceType):
    """
    Knight move logic.
    """

    @classmethod
    def move_yx_yx(cls, board, from_yx, to_yx, promote_to, has_my_king_moved,
                   dry_run=False):
        dy = abs(to_yx[0] - from_yx[0])
        dx = abs(to_yx[1] - from_yx[1])
        if sorted([dx, dy]) != [1, 2]:
            return False
        if cls.int2whose[board[to_yx]] == cls.mine:
            return False
        if not dry_run:
            board[to_yx] = board[from_yx]
            board[from_yx] = cls.space
        return True

    @classmethod
    def each_possible_move(cls, board, from_yx, has_my_king_moved):
        for off_y, off_x in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1),
                             (-1, -2), (1, -2), (2, -1)]:
            y = from_yx[0] + off_y
            x = from_yx[1] + off_x
            if 0 <= y < 8 and 0 <= x < 8 and \
                    cls.int2whose[board[y, x]] != cls.mine:
                yield y, x

    @classmethod
    def find_origin(cls, board, restrict_yx, to_yx):
        ret = []
        for off_y, off_x in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1),
                             (-1, -2), (1, -2), (2, -1)]:
            y = to_yx[0] + off_y
            x = to_yx[1] + off_x
            if 0 <= y < 8 and 0 <= x < 8 and board[y, x] == cls.my_knight and \
                    restrict_yx[0] in {None, y} and restrict_yx[1] in {None, x}:
                ret.append((y, x))
        return ret
