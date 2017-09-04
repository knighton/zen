import mxnet as mx


def embed(x, embeddings):
    vocab_size, channels = embeddings.shape
    channels_last = mx.nd.Embedding(
        data=x, weight=embeddings, input_dim=vocab_size, output_dim=channels,
        dtype=embeddings.dtype)
    axes = list(range(len(channels_last.shape)))
    axes = (axes[0],) + (axes[-1],) + tuple(axes[1:-1])
    return mx.nd.transpose(channels_last, axes)
