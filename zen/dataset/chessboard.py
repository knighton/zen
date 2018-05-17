from colorama import Fore, Style
import numpy as np


class Board(object):
    int2chr = '.prnbqkPRNBQK'

    chr2int = {}
    for i, c in enumerate(int2chr):
        chr2int[c] = i

    space, my_pawn, my_rook, my_knight, my_bishop, my_queen, my_king, \
        their_pawn, their_rook, their_knight, their_bishop, their_queen, \
        their_king = range(13)

    def __init__(self, arr):
        self.arr = arr

    def to_numpy(self):
        return self.arr

    @classmethod
    def from_numpy(cls, arr):
        return cls(arr)

    def to_text(self):
        lines = []
        for y in reversed(range(8)):
            line = []
            for x in range(8):
                n = self.arr[y, x]
                c = self.int2chr[n]
                line.append(c + ' ')
            lines.append(''.join(line))
        return ''.join(map(lambda line: line + '\n', lines))

    def to_color_text(self, heatmap, selected_yx):
        lines = []
        lines.append(Fore.WHITE + Style.DIM + '  ╔' + '─' * 17 + '╗' +
                     Style.RESET_ALL)
        for y in reversed(range(8)):
            if selected_yx is not None and y == selected_yx[0]:
                color = Fore.WHITE + Style.BRIGHT
            else:
                color = Fore.WHITE + Style.DIM
            line = [color + str(y + 1), '│' + Style.RESET_ALL]
            for x in range(8):
                n = self.arr[y, x]
                c = self.int2chr[n]
                if c == '.':
                    c = '.'  # '■'
                heat = heatmap[y][x]
                if selected_yx is not None and selected_yx[0] == y and \
                        selected_yx[1] == x:
                    color = Fore.WHITE
                elif heat < 0.001:
                    color = Style.DIM + Fore.BLUE
                elif heat <= 0.01:
                    color = Fore.BLUE
                elif heat < 0.025:
                    color = Style.BRIGHT + Fore.BLUE
                elif heat < 0.1:
                    color = Fore.CYAN
                elif heat < 0.2:
                    color = Style.BRIGHT + Fore.GREEN
                elif heat < 0.5:
                    color = Style.BRIGHT + Fore.YELLOW
                else:
                    color = Style.BRIGHT + Fore.RED
                line.append(color + c + Style.RESET_ALL)
            line.append(Fore.WHITE + Style.DIM + '│' + Style.RESET_ALL)
            lines.append(' '.join(line))
        left = Fore.WHITE + Style.DIM + '  ╚─' + Style.RESET_ALL
        middle = []
        for x in range(8):
            if selected_yx is not None and selected_yx[1] == x:
                color = Fore.WHITE + Style.BRIGHT
            else:
                color = Fore.WHITE + Style.DIM
            middle.append(color + '─' + Style.RESET_ALL)
        middle = (Fore.WHITE + Style.DIM + '─' + Style.RESET_ALL).join(middle)
        right = Fore.WHITE + Style.DIM + '─╝' + Style.RESET_ALL
        lines.append(left + middle + right)
        line = []
        for i, c in enumerate('abcdefgh'):
            if selected_yx is not None and i == selected_yx[1]:
                color = Fore.WHITE + Style.BRIGHT
            else:
                color = Fore.WHITE + Style.DIM
            line.append(color + c + Style.RESET_ALL)
        lines.append('    ' + ' '.join(line))
        return ''.join(map(lambda line: line + '\n', lines))

    @classmethod
    def from_text(cls, text):
        lines = text.strip().split('\n')

        def fix(s):
            s = s.strip()
            if ' ' in s:
                return s.split()

            if len(s) == 8:
                return list(s)

            if len(s[0]) == 8:
                return list(s[0])

            print('wtf', s)
            assert False

        lines = list(map(fix, lines))
        assert len(lines) == 8
        arr = np.zeros((8, 8), dtype='uint8')
        for y, line in enumerate(lines):
            assert len(line) == 8
            for x, c in enumerate(line):
                arr[8 - y - 1, x] = cls.chr2int[c]
        return cls(arr)

    @classmethod
    def initial(cls):
        return cls.from_text("""
            RNBQKBNR
            PPPPPPPP
            ........
            ........
            ........
            ........
            pppppppp
            rnbqkbnr
        """)

    def flip_color(self, n):
        if not n:
            return 0
        elif 1 <= n <= 6:
            return n + 6
        elif 7 <= n <= 12:
            return n - 6
        else:
            assert False

    def rotate(self):
        self.arr = np.rot90(np.rot90(self.arr))
        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                self.arr[i, j] = self.flip_color(self.arr[i, j])

    def rotate_coords(self, coords):
        if coords is None:
            return None
        x, y = coords
        if x is not None:
            x = 8 - x - 1
        if y is not None:
            y = 8 - y - 1
        return x, y

    def whose_piece(self, n):
        if not n:
            return 'space'
        elif 1 <= n <= 6:
            return 'mine'
        elif 7 <= n <= 12:
            return 'theirs'
        else:
            assert False

    def find_pieces(self, n, restrict):
        yy = restrict[0] if restrict[0] is not None else \
            range(self.arr.shape[0])
        xx = restrict[1] if restrict[1] is not None else \
            range(self.arr.shape[1])
        ret = []
        for i in yy:
            for j in xx:
                if self.arr[i, j] == n:
                    ret.append((i, j))
        return ret

    def find_origin_of_pawn_forward(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        if restrict_x not in {None, to_x}:
            return []

        y = to_y - 1
        x = to_x
        if self.arr[y, x] == self.my_pawn and restrict_y in {None, y}:
            return [(y, x)]
        elif to_y == 3 and self.arr[2, x] == self.space and \
                self.arr[1, x] == self.my_pawn and restrict_y in {None, 1}:
            return [(1, x)]
        else:
            return []

    def find_origin_of_pawn_en_passant_capture(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        if self.arr[to_y - 1, to_x] != self.their_pawn or \
                self.arr[to_y, to_x] != self.space:
            return []

        ret = []

        y = to_y - 1
        x = to_x - 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            coords = y, x
            ret.append((coords, True))

        y = to_y - 1
        x = to_x + 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            coords = y, x
            ret.append((coords, True))

        return ret

    def find_origin_of_pawn_normal_capture(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        y = to_y - 1
        x = to_x - 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        y = to_y - 1
        x = to_x + 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        return ret

    def find_origin_of_pawn(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to
        ret = []
        if self.arr[to] == self.space:
            ret += self.find_origin_of_pawn_forward(restrict, to)
            ret += self.find_origin_of_pawn_en_passant_capture(restrict, to)
        else:
            ret += self.find_origin_of_pawn_normal_capture(restrict, to)
        return ret

    def find_origin_of_rook(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to
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
                if self.arr[y, x] == self.my_rook:
                    ret.append((y, x))
                    break
                elif self.arr[y, x] != self.space:
                    break
        return list(set(ret))

    def find_origin_of_knight(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        offsets = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ]

        ret = []

        for off_y, off_x in offsets:
            y = to_y + off_y
            if not 0 <= y < 8:
                continue
            x = to_x + off_x
            if not 0 <= x < 8:
                continue
            n = self.arr[y, x]
            if n == self.my_knight:
                ret.append((y, x))

        if restrict_y is not None:
            ret = list(filter(lambda y_x: y_x[0] == restrict_y, ret))

        if restrict_x is not None:
            ret = list(filter(lambda y_x: y_x[1] == restrict_x, ret))

        return ret

    def find_origin_of_bishop(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        for off_y, off_x in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
            for i in range(8):
                y = to_y + i * off_y
                x = to_x + i * off_x
                if y == restrict_y or x == restrict_x:
                    continue
                if not 0 <= y < 8 or not 0 <= x < 8:
                    break
                n = self.arr[y, x]
                if n == self.my_bishop:
                    ret.append((y, x))

        return ret

    def find_origin_of_queen(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        for off_y, off_x in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1)]:
            for i in range(8):
                y = to_y + i * off_y
                x = to_x + i * off_x
                if y == restrict_y or x == restrict_x:
                    continue
                if not 0 <= y < 8 or not 0 <= x < 8:
                    break
                n = self.arr[y, x]
                if n == self.my_queen:
                    ret.append((y, x))

        return ret

    def find_origin_of_king(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        for off_y in [-1, 0, 1]:
            for off_x in [-1, 0, 1]:
                if off_y == off_x == 0:
                    continue
                y = to_y + off_y
                if not 0 <= y < 8:
                    continue
                x = to_x + off_x
                if not 0 <= x < 8:
                    continue
                n = self.arr[y, x]
                if n == self.my_king:
                    ret.append((y, x))

        if restrict_y is not None:
            ret = list(filter(lambda y, x: y == restrict_y, ret))

        if restrict_x is not None:
            ret = list(filter(lambda y, x: x == restrict_x, ret))

        return ret

    PIECE_CHR2FIND_ORIGIN = {
        'p': find_origin_of_pawn,
        'r': find_origin_of_rook,
        'n': find_origin_of_knight,
        'b': find_origin_of_bishop,
        'q': find_origin_of_queen,
        'k': find_origin_of_king,
    }

    def find_origin_of(self, piece_chr, maybe_from, to):
        if maybe_from is None:
            maybe_from = None, None
        elif maybe_from[0] is not None and maybe_from[1] is not None:
            assert False
        ret = self.PIECE_CHR2FIND_ORIGIN[piece_chr](self, maybe_from, to)
        assert len(ret) == 1
        ret = ret[0]
        if isinstance(ret[1], bool):
            assert len(ret) == 2
            return ret
        else:
            return ret, False

    def to_pgn_coords(self, y_x):
        y, x = y_x
        return 'abcdefgh'[x] + '12345678'[y]

    def white_queenside_castle(self):
        assert tuple(self.arr[0, :5]) == \
            (self.my_rook, self.space, self.space, self.space, self.my_king)
        self.arr[0, :5] = (self.space, self.space, self.my_king,
                           self.my_rook, self.space)

    def white_kingside_castle(self):
        assert tuple(self.arr[0, 4:]) == \
            (self.my_king, self.space, self.space, self.my_rook)
        self.arr[0, 4:] = (self.space, self.my_rook, self.my_king, self.space)

    def black_kingside_castle(self):
        assert tuple(self.arr[0, :4]) == \
            (self.my_rook, self.space, self.space, self.my_king)
        self.arr[0, :4] = (self.space, self.my_king, self.my_rook,
                           self.space)

    def black_queenside_castle(self):
        assert tuple(self.arr[0, 3:]) == \
            (self.my_king, self.space, self.space, self.space, self.my_rook)
        self.arr[0, 3:] = (self.space, self.my_rook, self.my_king,
                           self.space, self.space)

    def castle(self, from_, to, is_white):
        if is_white:
            assert from_ == (0, 4)
            if to == (0, 2):
                self.white_queenside_castle()
            elif to == (0, 6):
                self.white_kingside_castle()
            else:
                assert False
        else:
            assert from_ == (0, 3)
            if to == (0, 1):
                self.black_kingside_castle()
            elif to == (0, 5):
                self.black_queenside_castle()
            else:
                assert False

    def move(self, from_, to, is_white, en_passant, promote_to):
        if self.arr[from_[0], from_[1]] == self.my_king and \
                abs(from_[1] - to[1]) == 2:
            self.castle(from_, to, is_white)
        else:
            if promote_to is None:
                self.arr[to[0], to[1]] = self.arr[from_[0], from_[1]]
            else:
                self.arr[to[0], to[1]] = self.chr2int[promote_to]
            self.arr[from_[0], from_[1]] = self.space
            if en_passant:
                self.arr[to[0] - 1, to[1]] = self.space

    def apply_pgn_move(self, move, is_white):
        assert move not in {'0-1', '1-0', '1/2-1/2'}
        if move == 'kingside_castle':
            if is_white:
                from_, to = (0, 4), (0, 6)
            else:
                from_, to = (0, 3), (0, 1)
            en_passant = False
            promote_to = None
        elif move == 'queenside_castle':
            if is_white:
                from_, to = (0, 4), (0, 2)
            else:
                from_, to = (0, 3), (0, 5)
            en_passant = False
            promote_to = None
        else:
            piece, maybe_from, to, capture, promote_to = move
            piece = piece.lower()
            if promote_to is not None:
                promote_to = promote_to.lower()

            if not is_white:
                maybe_from = self.rotate_coords(maybe_from)
                to = self.rotate_coords(to)

            from_, en_passant = self.find_origin_of(piece, maybe_from, to)

            piece_at_target = self.arr[to[0], to[1]]
            whose = self.whose_piece(piece_at_target)
            if capture:
                if en_passant:
                    assert whose == 'space'
                else:
                    assert whose == 'theirs'
            else:
                assert whose == 'space'

        from_pgn = self.to_pgn_coords(from_)
        to_pgn = self.to_pgn_coords(to)
        top_line = '%s %s\n' % (from_pgn, to_pgn)
        ret = top_line + self.to_text() + '\n'

        self.move(from_, to, is_white, en_passant, promote_to)

        self.rotate()

        return ret
