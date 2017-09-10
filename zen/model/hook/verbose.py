import sys
from tqdm import tqdm

from .hook import Hook


class Verbose(Hook):
    def __init__(self, level):
        self.level = level

    def on_epoch_begin(self, progress, data):
        if self.level == 2:
            self.bar = None

    def on_train_batch_begin(self, progress, x, y_true):
        if self.level == 2:
            if self.bar is None:
                self.bar = tqdm(total=progress['num_batches'], leave=False)
            self.bar.update(1)

    def on_eval_batch_begin(self, progress, x, y_true):
        if self.level == 2:
            if self.bar is None:
                self.bar = tqdm(total=progress['num_batches'], leave=False)
            self.bar.update(1)

    def on_epoch_end(self, progress, result):
        if self.level == 2:
            self.bar.close()
        if self.level:
            for i in range(len(result['train'])):
                if i:
                    sys.stdout.write(' ' * 5)
                else:
                    epoch = result['progress']['epoch']
                    sys.stdout.write('%5d' % epoch)
                trains = result['train'][i]
                vals = result['val'][i]
                for j in range(len(trains)):
                    sys.stdout.write('  %.4f/%.4f' % (trains[j], vals[j]))
                sys.stdout.write('\n')


verbose = Verbose
