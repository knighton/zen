from .piece_type import PieceType


class Rook(PieceType):
    """
    Rook move logic.
    """

    @classmethod
    def move_yx_yx(cls, board, from_yx, to_yx, promote_to, has_my_king_moved,
                   dry_run=False):
        if from_yx[0] == to_yx[0]:
            if from_yx[1] < to_yx[1]:
                off = 0, 1
            else:
                off = 0, -1
        elif from_yx[1] == to_yx[1]:
            if from_yx[0] < to_yx[0]:
                off = 1, 0
            else:
                off = -1, 0
        else:
            return False
        for i in range(1, 8):
            y = from_yx[0] + i * off[0]
            x = from_yx[1] + i * off[1]
            if y == to_yx[0] and x == to_yx[1]:
                if cls.int2whose[board[y, x]] == cls.mine:
                    return False
                break
            else:
                if not board[y, x] == cls.space:
                    return False
        if not dry_run:
            board[to_yx] = board[from_yx]
            board[from_yx] = cls.space
        return True

    @classmethod
    def each_possible_move(cls, board, from_yx, has_my_king_moved):
        for off_y, off_x in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            for i in range(8):
                y = from_yx[0] + i * off_y
                x = from_yx[1] + i * off_x
                piece_type = board[y, x]
                if piece_type == cls.space:
                    yield y, x
                elif cls.int2whose[piece_type] == cls.theirs:
                    yield y, x
                    break
                else:
                    break
        return ret

    @classmethod
    def find_origin(cls, board, restrict_yx, to_yx):
        restrict_y, restrict_x = restrict_yx
        to_y, to_x = to_yx
        if restrict_y is not None:
            center = restrict_y, to_x
            offs = [(0, -1), (0, 1)]
        elif restrict_x is not None:
            center = to_y, restrict_x
            offs = [(-1, 0), (1, 0)]
        else:
            center = to_y, to_x
            offs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        ret = []
        for off_y, off_x in offs:
            for i in range(8):
                y = center[0] + i * off_y
                x = center[1] + i * off_x
                if y == to_y and x == to_x:
                    continue
                if not 0 <= y < 8 or not 0 <= x < 8:
                    break
                if board[y, x] == cls.my_rook:
                    ret.append((y, x))
                    break
                elif board[y, x] != cls.space:
                    break
        return list(set(ret))
