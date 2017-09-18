from ... import engine as E


"""
Embed ints in an n-dimensional float space.

    x, embeddings -> y

Input:
    x           variable (batch_size, shape...)
    embeddings  variable (vocab_size, channels)

Output:
    y           variable (batch_size, channels, shape...)
"""
embed = E.embed
