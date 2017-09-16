from .piece_type import PieceType


class Queen(PieceType):
    """
    Queen move logic.
    """

    @classmethod
    def move_yx_yx(cls, board, from_yx, to_yx, promote_to, has_my_king_moved,
                   dry_run=False):
        dy = to_yx[0] - from_yx[0]
        dx = to_yx[1] - from_yx[1]
        if dy == 0:
            if dx == 0:
                return False
            else:
                dx //= abs(dx)
        else:
            if dx == 0:
                dy //= abs(dy)
            else:
                if abs(dy) != abs(dx):
                    return False
                dy //= abs(dy)
                dx //= abs(dx)
        for i in range(1, 8):
            y = from_yx[0] + i * dy
            x = from_yx[1] + i * dx
            if (y, x) == to_yx:
                if cls.int2whose[board[y, x]] == cls.mine:
                    return False
                break
            else:
                if board[y, x] != cls.space:
                    return False
        if not dry_run:
            board[to_yx] = board[from_yx]
            board[from_yx] = cls.space
        return True

    @classmethod
    def each_possible_move(cls, board, from_yx, has_my_king_moved):
        from_y, from_x = from_yx
        for off_y, off_x in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1)]:
            for i in range(8):
                y = from_y + i * off_y
                x = from_x + i * off_x
                if not 0 <= y < 8 or not 0 <= x < 8:
                    break
                if cls.int2whose[board[y, x]] == cls.mine:
                    break
                yield y, x

    @classmethod
    def find_origin(cls, board, restrict_yx, to_yx):
        restrict_y, restrict_x = restrict_yx
        to_y, to_x = to_yx
        ret = []
        for off_y, off_x in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1)]:
            for i in range(1, 8):
                y = to_y + i * off_y
                x = to_x + i * off_x
                if not 0 <= y < 8 or not 0 <= x < 8:
                    break
                if board[y, x] == cls.space:
                    continue
                elif (restrict_y in {None, y} and restrict_x in {None, x} and
                      board[y, x] == cls.my_queen):
                    ret.append((y, x))
                    break
                else:
                    break
        return ret
