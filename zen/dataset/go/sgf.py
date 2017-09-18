class SGF(object):
    def __init__(self, fields, moves, black_first):
        self.fields = fields
        self.moves = moves
        self.black_first = black_first

    @classmethod
    def load(cls, text):
        text = text.strip()
        assert text.startswith('(;'), \
            'Text must start with `(;` (got `%s`).' % text[:2]
        assert text.endswith(')'), \
            'Text must end with `)` (got `%s`).' % text[-1]
        text = ']' + text[2:-1] + '['
        text = text.replace('(', '').replace(')', '')
        splits = []
        for i, c in enumerate(text):
            if c in {'[', ']'}:
                splits.append(i)
        pairs = []
        key = None
        values = []
        for i in range(len(splits) - 1):
            orig_subtext = text[splits[i] + 1 : splits[i + 1]]
            subtext = orig_subtext.strip()
            bracket_on_left = text[splits[i]]
            bracket_on_right = text[splits[i + 1]]
            if bracket_on_left == ']' and bracket_on_right == '[':
                if not subtext:
                    continue
                if key is not None:
                    pairs.append((key, values))
                key = subtext.replace('\n', '')
                values = []
            elif bracket_on_left == '[' and bracket_on_right == ']':
                values.append(subtext)
            else:
                assert False, 'Same-direction brackets (`%s%s%s`)' % \
                    (bracket_on_left, orig_subtext, bracket_on_right)
        if values:
            pairs.append((key, values))
        properties = {}
        moves = []
        black_first = None
        for key, value in pairs:
            if len(value) == 1:
                value, = value
            if key ==';B':
                if black_first is None:
                    black_first = True
                if black_first:
                    assert len(moves) % 2 == 0, 'Black moved twice.'
                else:
                    assert len(moves) % 2 == 1, 'Black moved twice.'
                if not value:
                    value = None
                moves.append(value)
            elif key == ';W':
                if black_first is None:
                    black_first = False
                if black_first:
                    assert len(moves) % 2 == 1, 'White moved twice.'
                else:
                    assert len(moves) % 2 == 0, 'White moved twice.'
                if not value:
                    value = None
                moves.append(value)
            else:
                if key in properties:
                    assert properties[key] == value, \
                        'Property `%s` set multiple times (`%s` vs `%s`).' % \
                        (key, properties[key], value)
                else:
                    properties[key] = value
        return cls(properties, moves, black_first)
