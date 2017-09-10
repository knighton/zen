from .hook import Hook


class Stop(Hook):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs

    def on_train_begin(self, model, train_kwargs):
        epoch_begin = train_kwargs['epoch_begin']
        return epoch_begin + self.num_epochs

    def on_epoch_begin(self, progress, data):
        if progress['epoch'] == progress['epoch_begin'] + self.num_epochs:
            return True

    def on_epoch_end(self, progress, result):
        if progress['epoch'] == \
                progress['epoch_begin'] + self.num_epochs - 1:
            return True


stop = Stop
