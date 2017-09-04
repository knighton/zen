from .transform import Transform


class Lower(Transform):
    def transform(self, x):
        rrr = []
        for line in x:
            if isinstance(line, str):
                rr = line.lower()
            else:
                rr = []
                for token in line:
                    rr.append(token.lower())
            rrr.append(rr)
        return rrr

    def inverse_transform(x):
        return x
