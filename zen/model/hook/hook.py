class Hook(object):
    def on_train_begin(self, model, train_kwargs):
        """
        -> epoch end_excl, if it exists (used for progress bar)
        """
        return None

    def on_epoch_begin(self, progress, data):
        """
        -> whether to stop training
        """
        return False

    def on_train_batch_begin(self, progress, x, y_true):
        """
        -> whether to stop training
        """
        return False

    def on_train_batch_end(self, progress, result):
        """
        -> whether to stop training
        """
        return False

    def on_eval_batch_begin(self, progress, x, y_true):
        """
        -> whether to stop training
        """
        return False

    def on_eval_batch_end(self, progress, result):
        """
        -> whether to stop training
        """
        return False

    def on_epoch_end(self, progress, result):
        """
        -> whether to stop training
        """
        return False

    def on_train_end(self, results):
        """
        ->
        """
        pass
