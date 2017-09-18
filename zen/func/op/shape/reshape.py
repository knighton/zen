from ... import core as C


"""
x          tensor (batch_size, in_shape...)  Data to reshape.
out_shape  shape...                          Shape without batch_size dim.
"""
reshape = C.reshape
