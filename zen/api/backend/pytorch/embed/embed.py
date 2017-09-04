def embed_pytorch(x, embeddings):
    channels_last = embeddings.index_select(0, x.view(-1))
    channels_last = channels_last.view(x.size() + (-1,))
    axes = list(range(channels_last.dim()))
    axes = (axes[0],) + (axes[-1],) + tuple(axes[1:-1])
    return channels_last.permute(*axes).contiguous()
