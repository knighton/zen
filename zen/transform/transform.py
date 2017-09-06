_OFFSET = 4


class Transform(object):
    def start_pipe(self, verbose, depth):
        if verbose:
            if not depth:
                print('%s+%s' % (' ' * _OFFSET, '-' * 10))
            print('%s| %s%s:' % (' ' * _OFFSET, '  ' * depth,
                                 self.__class__.__name__))

    def done(self, t, verbose, depth):
        if verbose:
            print('%s| %s%s (%.3f sec)' %
                  (' ' * _OFFSET, '  ' * depth, self.__class__.__name__, t))
            if self.__class__.__name__ == 'Pipe' and not depth:
                print('%s+%s' % (' ' * _OFFSET, '-' * 10))

    def fit(self, x, verbose, depth):
        pass

    def transform(self, x, verbose=0, depth=0):
        return x

    def fit_transform(self, x, verbose=0, depth=0):
        self.fit(x, verbose, depth)
        return self.transform(x, verbose, depth)

    def inverse_transform(self, x):
        raise NotImplementedError
