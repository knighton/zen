class PGN(object):
    last_moves = {'0-1', '1/2-1/2', '1-0', '*'}

    def __init__(self, tags, moves):
        self.tags = tags
        self.moves = moves

    @classmethod
    def without_comments(cls, text):
        while '{' in text:
            a = text.find('{')
            z = text.find('}')
            assert a < z
            text = text[:a] + text[z + 1:]
        return text

    @classmethod
    def from_text(cls, text):
        text = cls.without_comments(text)

        lines = text.split('\n')

        tags = {}
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                break
            assert line.startswith('[')
            assert line.endswith('"]')
            x = line.index(' "')
            key = line[1:x]
            value = line[x + 2:-2]
            assert key not in tags
            tags[key] = value

        ss = []
        for line in lines[i:]:
            ss += line.replace('.', ' . ').split()

        z = len(ss) % 4
        if z == 0:
            pass
        elif z == 1:
            assert ss[-1] in cls.last_moves, text
        else:
            assert False, text

        moves = []
        for i in range(len(ss) // 4):
            assert int(ss[i * 4]) == i + 1
            assert ss[i * 4 + 1] == '.'
            white = ss[i * 4 + 2]
            moves.append(white)
            black = ss[i * 4 + 3]
            moves.append(black)

        if z == 1:
            moves.append(ss[-1])

        assert moves[-1] in cls.last_moves

        return PGN(tags, moves)
