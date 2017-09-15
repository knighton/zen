from .piece_type import PieceType


class King(PieceType):
    """
    King move logic.
    """

    @classmethod
    def black_kingside_castle(cls, board):
        if tuple(board[0, :4]) != \
                (cls.my_rook, cls.space, cls.space, cls.my_king):
            return False
        board[0, :4] = cls.space, cls.my_king, cls.my_rook, cls.space
        return True

    @classmethod
    def black_queenside_castle(cls, board):
        if tuple(board[0, 3:]) != (cls.my_king, cls.space, cls.space, cls.space,
                                   cls.my_rook):
            return False
        board[0, 3:] = cls.space, cls.my_rook, cls.my_king, cls.space, cls.space
        return True

    @classmethod
    def white_queenside_castle(cls, board):
        if tuple(board[0, :5]) != (cls.my_rook, cls.space, cls.space, cls.space,
                                   cls.my_king):
            return False
        board[0, :5] = cls.space, cls.space, cls.my_king, cls.my_rook, cls.space
        return True

    @classmethod
    def white_kingside_castle(cls, board):
        if tuple(board[0, 4:]) != (cls.my_king, cls.space, cls.space,
                                   cls.my_rook):
            return False
        board[0, 4:] = cls.space, cls.my_rook, cls.my_king, cls.space
        return True

    @classmethod
    def castle(cls, board, from_yx, to_yx, promote_to, has_my_king_moved):
        if has_my_king_moved:
            return False
        if from_yx == (0, 3):
            if to_yx == (0, 1):
                return cls.black_kingside_castle(board)
            elif to_yx == (0, 5):
                return cls.black_queenside_castle(board)
            else:
                return False
        elif from_yx == (0, 4):
            if to_yx == (0, 2):
                return cls.white_queenside_castle(board)
            elif to_yx == (0, 6):
                return cls.white_kingside_castle(board)
            else:
                return False
        else:
            return False

    @classmethod
    def move_yx_yx(cls, board, from_yx, to_yx, promote_to, has_my_king_moved,
                   dry_run=False):
        dy = to_yx[0] - from_yx[0]
        dx = to_yx[1] - from_yx[1]
        if from_yx[0] == 0 and abs(dx) == 2:
            return cls.castle(board, from_yx, to_yx, has_my_king_moved, dry_run)
        if cls.int2whose[board[to_yx]] == cls.mine:
            return False
        if abs(dy) == 1:
            if abs(dx) not in {0, 1}:
                return False
        elif abs(dy) == 0:
            if abs(dx) != 1:
                return False
        else:
            return False
        if not dry_run:
            board[to_yx] = board[from_yx]
            board[from_yx] = cls.space
        return True

    @classmethod
    def each_possible_move(cls, board, from_yx, has_my_king_moved):
        for off_y, off_x in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1)]:
            y = from_yx[0] + off_y
            x = from_yx[1] + off_x
            if not 0 <= y < 8 or not 0 <= x < 8:
                continue
            if cls.int2whose[board[y, x]] == cls.mine:
                continue
            yield y, x

    @classmethod
    def find_origin(cls, board, restrict_yx, to_yx):
        restrict_y, restrict_x = restrict_yx
        ret = []
        for off_y, off_x in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1)]:
            y = to_yx[0] + off_y
            x = to_yx[1] + off_x
            if not 0 <= y < 8 or not 0 <= x < 8:
                continue
            if restrict_y not in {None, y} or restrict_x not in {None, x}:
                continue
            if board[y, x] != cls.my_king:
                continue
            ret.append((y, x))
        return ret
