from colorama import Back, Fore, Style
import numpy as np

from .kgs import load_kgs


class GoGame(object):
    int2chr = 'ABCDEFGHJKLMNOPQRST'
    int2full_chr = ''.join(map(chr, list(range(0xFF21, 0xFF29)) +
                                    list(range(0xFF2a, 0xFF35))))

    black = 0
    space = 1
    white = 2

    def __init__(self, board):
        self.board = board

    @classmethod
    def aa_to_yx(cls, aa):
        if aa is None:
            return None
        x = ord(aa[0]) - ord('a')
        y = ord(aa[1]) - ord('a')
        return y, x

    @classmethod
    def aa_to_a1(cls, aa):
        if aa is None:
            return None
        x = ord(aa[0]) - ord('a')
        y = ord(aa[1]) - ord('a')
        return cls.int2chr[x], y + 1

    def move(self, color, yx):
        self.board[yx] = color
        return True

    def to_training_data(self, color, yx):
        shape = self.board.shape
        assert shape[0] <= 19
        assert shape[1] <= 19
        if color == self.black:
            board = self.board
        elif color == self.white:
            board = (self.board - 1) * -1 + 1
        else:
            assert False
        ints = []
        for y in range(shape[0]):
            int_ = 0
            for x in reversed(range(shape[1])):
                int_ *= 3
                int_ += board[y, x]
            ints.append(int_)
        y, x = yx
        ints.append(y * shape[0] + x)
        return np.array(ints, dtype='uint32')

    board_bg = ''

    waxing_moon = chr(0x1F312)

    black_stone = Style.BRIGHT + Fore.BLACK + board_bg + waxing_moon + \
        Style.RESET_ALL

    free_cross = Style.BRIGHT + Fore.BLACK + board_bg + chr(0xFF0B) + \
        Style.RESET_ALL

    free_dot = Style.BRIGHT + Fore.BLACK + board_bg + chr(0x30FB) + Style.RESET_ALL

    white_stone = Style.BRIGHT + Fore.WHITE + board_bg + waxing_moon + \
        Style.RESET_ALL

    def to_human(self, heatmap=None, selected_yx=None):
        indent = ' ' * 4
        shape = self.board.shape
        lines = []
        for y in reversed(range(shape[0])):
            on_left = '%s%2d%s ' % (Style.BRIGHT, y + 1, Style.RESET_ALL)
            row = []
            for x in range(shape[1]):
                n = self.board[y, x]
                if n == self.black:
                    square = self.black_stone
                elif n == self.space:
                    is_mark = shape == (19, 19) and y in {3, 9, 15} and \
                        x in {3, 9, 15}
                    square = self.free_cross if is_mark else self.free_dot
                elif n == self.white:
                    square = self.white_stone
                else:
                    assert False
                row.append(square)
            line = indent + on_left + ''.join(row)
            lines.append(line)
        cc = []
        for i in range(shape[1]):
            c = self.int2full_chr[i]
            cc.append(c)
        line = indent + '   ' + ''.join(cc) + Style.RESET_ALL
        lines.append(line)
        return '\n'.join(lines)

    @classmethod
    def sgf_to_initial_board(cls, sgf):
        size = int(sgf.fields['SZ'])
        assert size == 19
        board = np.ones((19, 19), dtype='int8') * cls.space
        for aa in sgf.fields.get('AB', []):
            board[cls.aa_to_yx(aa)] = cls.black
        for aa in sgf.fields.get('AW', []):
            board[cls.aa_to_yx(aa)] = cls.white
        return board

    @classmethod
    def replay_sgf(cls, sgf):
        board = cls.sgf_to_initial_board(sgf)
        game = GoGame(board)
        arrs = []
        for i, aa in enumerate(sgf.moves):
            print(cls.aa_to_a1(aa))
            if aa is None:
                continue
            print(game.to_human())
            print()
            print()
            yx = cls.aa_to_yx(aa)
            is_black = i % 2 == int(not sgf.black_first)
            color = cls.black if is_black else cls.white
            arr = game.to_training_data(color, yx)
            assert game.move(color, yx)
            arrs.append(arr)
        print(game.to_human())
        print()
        print()
        return np.stack(arrs)


def load_go(verbose=2):
    sgfs = []
    for sgf in load_kgs(verbose):
        if not sgf.moves:
            continue
        sgfs.append(sgf)
        if len(sgfs) == 100:
            break
    np.random.shuffle(sgfs)
    arrs = []
    for sgf in sgfs:
        arr = GoGame.replay_sgf(sgf) 
        arrs.append(arr)
    assert False
