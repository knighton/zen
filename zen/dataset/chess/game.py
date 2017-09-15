import numpy as np

from .bishop import Bishop
from .king import King
from .knight import Knight
from .pawn import Pawn
from .piece_type import PieceType as Chess
from .queen import Queen
from .rook import Rook


class Game(object):
    """
    A chess game.
    """

    piece_classes = Pawn, Rook, Knight, Bishop, Queen, King

    def __init__(self, board, moves, has_my_king_moved, has_their_king_moved,
                 i_won):
        self.board = board
        self.moves = moves
        self.has_my_king_moved = has_my_king_moved
        self.has_their_king_moved = has_their_king_moved
        self.i_won = i_won

    @classmethod
    def from_text(cls, text, moves, has_my_king_moved, has_their_king_moved,
                  i_won):
        text = ''.join(text.split())
        assert len(text) == 64
        arr = np.zeros((64,), dtype='int8')
        for i, c in enumerate(text):
            arr[i] = Chess.chr2int[c]
        board = arr.reshape((8, 8))
        board = np.flip(board, 0)
        return cls(board, moves, has_my_king_moved, has_their_king_moved, i_won)

    @classmethod
    def start(cls):
        board = """
            RNBQKBNR
            PPPPPPPP
            ........
            ........
            ........
            ........
            pppppppp
            rnbqkbnr
        """
        return cls.from_text(board, [], False, False, None)

    @classmethod
    def a1_to_yx(self, a1):
        assert len(a1) == 2
        assert 'a' <= a1[0] <= 'h'
        assert '1' <= a1[1] <= '8'
        x = ord(a1[0]) - ord('a')
        y = ord(a1[1]) - ord('1')
        return y, x

    @classmethod
    def decode_pgn_move(cls, text):
        assert text not in {'0-1', '1-0', '1/2-1/2', '*'}

        if text.endswith('+') or text.endswith('#'):
            text = text[:-1]

        if text == 'O-O':
            return 'kingside_castle'

        if text == 'O-O-O':
            return 'queenside_castle'

        if 'x' in text:
            capture = True
            text = text.replace('x', '')
        else:
            capture = False

        if '=' in text:
            eq = text.find('=')
            promote_to = text[eq + 1:]
            text = text[:eq]
        else:
            promote_to = None

        try:
            move_to = cls.a1_to_yx(text[-2:])
        except:
            raise

        c = text[:-2]
        move_from = None, None
        if not len(c):
            piece = 'P'
        elif len(c) == 1:
            if c.isupper():
                piece = c
            elif c.islower():
                piece = 'P'
                x = ord(c) - ord('a')
                move_from = None, x
            else:
                assert False
        elif len(c) == 2:
            if c[0].isupper() and c[1].islower():
                piece = c[0]
                x = ord(c[1]) - ord('a')
                move_from = None, x
            elif c[0].isupper() and c[1].isdigit():
                piece = c[0]
                y = ord(c[1]) - ord('1')
                move_from = y, None
            else:
                assert False
        else:
            assert False

        piece = Chess.chr2int[piece.lower()]
        if promote_to is not None:
            promote_to = Chess.chr2int[promote_to.lower()]
        return piece, move_from, move_to, capture, promote_to

    @classmethod
    def switch_color(cls, n):
        if not n:
            return 0
        elif 1 <= n <= 6:
            return n + 6
        elif 7 <= n <= 12:
            return n - 6
        else:
            assert False

    def switch_sides(self):
        self.board = np.rot90(np.rot90(self.board))
        for i in range(64):
            piece = self.board[i // 8, i % 8]
            self.board[i // 8, i % 8] = self.switch_color(piece)
        b = self.has_my_king_moved
        self.has_my_king_moved = self.has_their_king_moved
        self.has_their_king_moved = b

    @classmethod
    def rotate_coords(cls, coords):
        y, x = coords
        if x is not None:
            x = 8 - x - 1
        if y is not None:
            y = 8 - y - 1
        return y, x

    def apply_yx_yx_move(self, from_yx, to_yx, promote_to):
        assert 0 <= from_yx[0] < 8
        assert 0 <= from_yx[1] < 8
        assert 0 <= to_yx[0] < 8
        assert 0 <= to_yx[1] < 8
        if promote_to is not None:
            assert (0 <= promote_to < len(Chess.int2whose) and
                    Chess.int2whose[promote_to] == Chess.mine)
        assert Chess.int2whose[self.board[from_yx]] == Chess.mine
        class_ = self.piece_classes[self.board[from_yx] - 1]
        ret = class_.move_yx_yx(self.board, from_yx, to_yx, promote_to,
                                self.has_my_king_moved)
        self.switch_sides()
        return ret

    def find_my_king(self):
        for y in range(8):
            for x in range(8):
                if self.board[y, x] == Chess.my_king:
                    return y, x

    def is_piece(self, y, x, piece):
        if not 0 <= y < 8:
            return False
        if not 0 <= x < 8:
            return False
        return self.board[y, x] == piece

    def is_in_check(self):
        king_y, king_x = self.find_my_king()

        if self.is_piece(king_y + 1, king_x - 1, Chess.their_pawn):
            return True
        if self.is_piece(king_y + 1, king_x + 1, Chess.their_pawn):
            return True

        for off_y, off_x in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1),
                             (-1, -2), (1, -2), (2, -1)]:
            if self.is_piece(king_y + off_y, king_x + off_x,
                             Chess.their_knight):
                return True

        for off_y, off_x in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            for i in range(1, 8):
                y = king_y + i * off_y
                x = king_x + i * off_x
                if self.is_piece(y, x, Chess.space):
                    continue
                elif self.is_piece(y, x, Chess.their_rook):
                    return True
                elif self.is_piece(y, x, Chess.their_queen):
                    return True
                else:
                    break

        for off_y, off_x in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            for i in range(1, 8):
                y = king_y + i * off_y
                x = king_x + i * off_x
                if self.is_piece(y, x, Chess.space):
                    continue
                elif self.is_piece(y, x, Chess.their_bishop):
                    return True
                elif self.is_piece(y, x, Chess.their_queen):
                    return True
                else:
                    break

        for off_y, off_x in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1)]:
            if self.is_piece(king_y + off_y, king_x + off_x, Chess.their_king):
                return True

        return False

    def find_origin_of_piece(self, piece_type, restrict_yx, to_yx):
        class_ = self.piece_classes[piece_type - 1]
        options = class_.find_origin(self.board, restrict_yx, to_yx)
        filtered = []
        if len(options) != 1:
            for from_yx in options:
                to_piece = self.board[to_yx]
                self.board[to_yx] = self.board[from_yx]
                self.board[from_yx] = Chess.space
                if not self.is_in_check():
                    filtered.append(from_yx)
                self.board[from_yx] = self.board[to_yx]
                self.board[to_yx] = to_piece
            ret, = filtered
        else:
            ret, = options
        return ret

    def apply_pgn_move(self, move, is_white):
        move = self.decode_pgn_move(move)
        if move == 'kingside_castle':
            if is_white:
                from_yx = 0, 4
                to_yx = 0, 6
            else:
                from_yx = 0, 3
                to_yx = 0, 1
            promote_to = None
        elif move == 'queenside_castle':
            if is_white:
                from_yx = 0, 4
                to_yx = 0, 2
            else:
                from_yx = 0, 3
                to_yx = 0, 5
            promote_to = None
        else:
            piece_type, restrict_yx, to_yx, is_capture, promote_to = move
            class_ = self.piece_classes[piece_type - 1]
            if not is_white:
                restrict_yx = self.rotate_coords(restrict_yx)
                to_yx = self.rotate_coords(to_yx)
            from_yx = self.find_origin_of_piece(piece_type, restrict_yx, to_yx)
            assert class_.is_capture(self.board, from_yx, to_yx) == is_capture
        ok = self.apply_yx_yx_move(from_yx, to_yx, promote_to)
        return ok, from_yx, to_yx

    @classmethod
    def each_board(cls, pgn):
        game = Game.start()
        result = pgn.tags['Result']
        will_win = {
            '1-0': 1,
            '1/2-1/2': 0,
            '0-1': -1,
        }[result]
        for i, move in enumerate(pgn.moves[:-1]):
            is_white = i % 2 == 0
            ok, from_yx, to_yx = game.apply_pgn_move(move, is_white)
            assert ok
            will_win *= -1
        return []  # XXX

    def can_move_piece_at(self, from_yx):
        """
        Return whether the player can move the piece.
        """
        piece_type = self.board[from_yx]
        assert Chess.int2whose[piece_type] == Chess.mine
        return self.piece_classes[piece_type - 1].can_move(
            self.board, from_yx, self.has_my_king_moved)

    def get_movable_pieces(self):
        """
        Return an 8x8 int8 grid where the ones are the squares of pieces that
        the player can move.

        Used to filter piece selection candidates to just the legal ones.
        """
        ret = np.zeros((8, 8), dtype='int8')
        for y in range(8):
            for x in range(8):
                piece_type = self.board[y, x]
                if Chess.int2whose[piece_type] != Chess.mine:
                    continue
                ret[y, x] = self.can_move_piece_at((y, x))
        return ret

    def get_possible_moves(self, from_yx):
        """
        Return an 8x8 int8 grid where the ones are the squares where the
        player's selected piece can move.  If the piece can't move, returns all
        zeros.  If the square actually contains a space or opponent's piece,
        raises an exception.

        Used to filter target selection candidates.
        """
        piece_type = self.board[from_yx]
        assert Chess.int2whose[piece_type] == Chess.mine
        class_ = self.piece_classes[piece_type - 1]
        return class_.get_possible_moves(
            self.board, from_yx, self.has_my_king_moved)
