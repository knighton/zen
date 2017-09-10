from argparse import ArgumentParser

from zen.dataset.cover_type import load_cover_type
from zen.layer import *
from zen.transform.one_hot import one_hot


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def mlp(num_features, num_classes):
    layer = lambda n: Dense(n) > BatchNorm > ReLU > Dropout(0.5) > Z
    mlp = layer(64) > layer(64) > layer(64) > layer(64) > Z
    in_shape = num_features,
    return Input(in_shape) > mlp > Dense(num_classes) > Softmax > Z


def observe(x_train, column_index):
    column = x_train[:, column_index]
    return column.mean(), column.std()


def scale(x, column_index, column_mean, column_std):
    x[:, column_index] -= column_mean
    x[:, column_index] /= column_std


class CoverTypeTransformer(object):
    def fit(self, x_train, verbose):
        self.elev_mean, self.elev_std = observe(x_train, 0)
        if verbose:
            print('Elevation: mean %.3f, std %.3f meters.' %
                  (self.elev_mean, self.elev_std))

        self.aspect_mean, self.aspect_std = observe(x_train, 1)
        if verbose:
            print('Aspect: mean %.3f, std %.3f degrees azimuth.' %
                  (self.aspect_mean, self.aspect_std))

        self.slope_mean, self.slope_std = observe(x_train, 2)
        if verbose:
            print('Slope: mean %.3f, std %.3f degrees.' %
                  (self.slope_mean, self.slope_std))

        self.water_horz_dist_mean, self.water_horz_dist_std = \
            observe(x_train, 3)
        if verbose:
            print(('Horizontal distance to hydrology: mean %.3f, std %.3f ' +
                   'meters.') % (self.water_horz_dist_mean,
                                 self.water_horz_dist_std))

        self.water_vert_dist_mean, self.water_vert_dist_std = \
            observe(x_train, 4)
        if verbose:
            print('Vertical distance to hydrology: mean %.3f, std %.3f meters.'
                  % (self.water_vert_dist_mean, self.water_vert_dist_std))

        self.road_dist_mean, self.road_dist_std = observe(x_train, 5)
        if verbose:
            print(('Horizontal distance to roadways: mean %.3f, std %.3f ' +
                   'meters.') % (self.road_dist_mean, self.road_dist_std))

        self.fire_dist_mean, self.fire_dist_std = observe(x_train, 9)
        if verbose:
            print(('Horizontal distance to fire points: mean %.3f, std %.3f ' +
                   'meters.') % (self.fire_dist_mean, self.fire_dist_std))

    def transform(self, x, verbose):
        y = x.astype('float32')
        scale(y, 0, self.elev_mean, self.elev_std)
        scale(y, 1, self.aspect_mean, self.aspect_std)
        scale(y, 2, self.slope_mean, self.slope_std)
        scale(y, 3, self.water_horz_dist_mean, self.water_horz_dist_std)
        scale(y, 4, self.water_vert_dist_mean, self.water_vert_dist_std)
        scale(y, 5, self.road_dist_mean, self.road_dist_std)
        y[:, 6] /= 127.5
        y[:, 6] -= 1.
        y[:, 7] /= 127.5
        y[:, 7] -= 1.
        y[:, 8] /= 127.5
        y[:, 8] -= 1.
        scale(y, 9, self.fire_dist_mean, self.fire_dist_std)
        return y

    def fit_transform(self, x_train, verbose):
        self.fit(x_train, verbose)
        return self.transform(x_train, verbose)


def run(args):
    data = load_cover_type(args.val_frac, args.verbose)
    transformer = CoverTypeTransformer()
    x_train = transformer.fit_transform(data[0][0], args.verbose)
    x_val = transformer.transform(data[1][0], args.verbose)
    num_classes = data[0][1].max() + 1
    y_train = one_hot(data[0][1], num_classes)
    y_val = one_hot(data[1][1], num_classes)
    data = (x_train, y_train), (x_val, y_val)
    num_features = data[0][0].shape[1]
    num_classes = data[0][1].shape[1]
    if args.verbose:
        print('Train: %s -> %s (mean %.3f, std %.3f).' %
              (data[0][0].shape, data[0][1].shape, data[0][0].mean(),
               data[0][0].std()))
        print('Val:   %s -> %s (mean %.3f, std %.3f).' %
              (data[1][0].shape, data[1][1].shape, data[1][0].mean(),
               data[1][0].std()))
    model = mlp(num_features, num_classes)
    model.train_classifier(data, opt=args.opt, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
