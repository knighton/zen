class PieceType(object):
    """
    A type of chess piece (pawn, rook, etc.), containing the game logic
    associated with that piece.
    """
    space, my_pawn, my_rook, my_knight, my_bishop, my_queen, my_king, \
    their_pawn, their_rook, their_knight, their_bishop, their_queen, \
    their_king = range(13)

    int2chr = '.PRNBQKprnbqk'

    chr2int = {}
    for i, c in enumerate(int2chr):
        chr2int[c] = i

    mine = 1
    theirs = -1
    int2whose = [space] + [mine] * 6 + [theirs] * 6

    @classmethod
    def move_yx_yx(cls, board, from_yx, to_yx, promote_to, has_my_king_moved,
                   dry_run=False):
        raise NotImplementedError

    @classmethod
    def each_possible_move(cls, board, from_yx, has_my_king_moved):
        raise NotImplementedError

    @classmethod
    def get_possible_moves(cls, board, from_yx, has_my_king_moved):
        ret = np.zeros((8, 8), dtype='int8')
        for to_y in cls.each_possible_move(board, from_yx, has_my_king_moved):
            ret[to_y] = 1
        return ret

    @classmethod
    def can_move(cls, board, from_yx, has_my_king_moved):
        for to_yx in cls.each_possible_move(board, from_yx, has_my_king_moved):
            return True
        return False

    @classmethod
    def find_origin(cls, board, restrict_yx, to_yx):
        raise NotImplementedError

    @classmethod
    def is_capture(cls, board, from_yx, to_yx):
        """
        Overridden by pawns for en passant.
        """
        return cls.int2whose[board[to_yx]] == cls.theirs
