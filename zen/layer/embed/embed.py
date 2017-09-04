from ... import api as Z
from ... import init
from ..layer import Layer, Spec, Sugar


class EmbedLayer(Layer):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = self.add_param(embeddings)

    def forward(self, x, is_training):
        return Z.embed(x, self.embeddings)


class EmbedSpec(Spec):
    def __init__(self, vocab_size, channels, dtype=None,
                 embeddings_init='uniform'):
        super().__init__()
        assert Z.is_dim(vocab_size)
        assert Z.is_dim(channels)
        self.vocab_size = vocab_size
        self.channels = channels
        self.dtype = dtype if dtype else Z.floatx()
        self.embeddings_init = init.get(embeddings_init)

    def build(self, in_shape, in_dtype):
        assert Z.is_shape(in_shape, 1)
        assert in_dtype == 'int64'
        in_len, = in_shape
        embeddings_shape = self.vocab_size, self.channels
        embeddings = self.embeddings_init(embeddings_shape, self.dtype)
        out_shape = self.channels, in_len
        return EmbedLayer(embeddings), out_shape, self.dtype


Embed = Sugar(EmbedSpec)
