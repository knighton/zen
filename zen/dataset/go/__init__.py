from colorama import Back, Fore, Style
import numpy as np

from .kgs import load_kgs


class GoGame(object):
    int2chr = 'ABCDEFGHJKLMNOPQRST'

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

    def to_human(self, heatmap=None, selected_yx=None):
        indent = ' ' * 4
        shape = self.board.shape
        lines = []
        cc = []
        line = indent + '    ' + ' '.join(cc)
        lines.append(line)
        line = Style.DIM + indent + '   ╔─' + '─' * (shape[1] * 2 - 1) + \
            '─╗' + Style.RESET_ALL
        lines.append(line)
        for y in reversed(range(shape[0])):
            ss = []
            ss.append('%s%2d' % (Style.DIM, y + 1))
            ss.append('│%s' % Style.RESET_ALL)
            for x in range(shape[1]):
                n = self.board[y, x]
                if n == self.black:
                    square = Style.DIM + Fore.WHITE + chr(0x23FA) + \
                        Style.RESET_ALL
                elif n == self.space:
                    is_mark = shape == (19, 19) and y in {3, 9, 15} and \
                        x in {3, 9, 15}
                    c = '+' if is_mark else '᛫'
                    square = Style.DIM + Fore.WHITE + c + Style.RESET_ALL
                elif n == self.white:
                    square = Style.BRIGHT + Fore.WHITE + chr(0x23FA) + \
                        Style.RESET_ALL
                else:
                    assert False
                ss.append(square)
            ss.append(Style.DIM + '│' + Style.RESET_ALL)
            line = indent + ' '.join(ss)
            lines.append(line)
        line = Style.DIM + indent + '   ╚─' + '─' * (shape[1] * 2 - 1) + \
            '─╝' + Style.RESET_ALL
        lines.append(line)
        cc = []
        for i in range(shape[1]):
            c = self.int2chr[i]
            cc.append(c)
        line = Style.DIM + indent + '     ' + ' '.join(cc) + Style.RESET_ALL
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
