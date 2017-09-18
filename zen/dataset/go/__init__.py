from colorama import Back, Fore, Style
import numpy as np

from .kgs import load_kgs


class GoGame(object):
    int2chr = 'ABCDEFGHJKLMNOPQRST'

    def __init__(self, board):
        self.board = board

    @classmethod
    def aa_to_yx(cls, aa):
        x = ord(aa[0]) - ord('a')
        y = ord(aa[1]) - ord('a')
        return y, x

    def move(self, yx):
        self.board[yx] = 1
        return True

    def switch_sides(self):
        self.board *= -1

    def to_training_data(self, yx):
        shape = self.board.shape
        assert shape[0] <= 19
        assert shape[1] <= 19
        ints = []
        for y in range(shape[0]):
            int_ = 0
            for x in reversed(range(shape[1])):
                int_ *= 3
                int_ += self.board[y, x] + 1
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
                mark = shape == (19, 19) and y in {3, 9, 15} and x in {3, 9, 15}
                s = {
                    -1: Style.BRIGHT + Fore.WHITE + chr(0x23FA) + \
                        Style.RESET_ALL,
                    0: Style.DIM + Fore.WHITE + ('+' if mark else '᛫') + \
                        Style.RESET_ALL,
                    1: Style.DIM + Fore.WHITE + chr(0x23FA) + Style.RESET_ALL,
                }[self.board[y, x]]
                ss.append(s)
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
        board = np.zeros((19, 19), dtype='int8')
        if sgf.black_first:
            for aa in sgf.fields.get('AB', []):
                board[cls.aa_to_yx(aa)] = 1
            for aa in sgf.fields.get('AW', []):
                board[cls.aa_to_yx(aa)] = -1
        else:
            for aa in sgf.fields.get('AB', []):
                board[cls.aa_to_yx(aa)] = -1
            for aa in sgf.fields.get('AW', []):
                board[cls.aa_to_yx(aa)] = 1
        return board

    @classmethod
    def replay_sgf(cls, sgf):
        board = cls.sgf_to_initial_board(sgf)
        game = GoGame(board)
        arrs = []
        for aa in sgf.moves:
            if aa is None:
                game.switch_sides()
                continue
            print(game.to_human())
            print()
            print()
            yx = cls.aa_to_yx(aa)
            arr = game.to_training_data(yx)
            assert game.move(yx)
            game.switch_sides()
            arrs.append(arr)
        print(game.to_human())
        print()
        print()
        return np.stack(arrs)


def load_go(verbose=2):
    sgfs = []
    for sgf in load_kgs(verbose):
        sgfs.append(sgf)
        if len(sgfs) == 100:
            break
    np.random.shuffle(sgfs)
    arrs = []
    for sgf in sgfs:
        arr = GoGame.replay_sgf(sgf) 
        arrs.append(arr)
    assert False
