from .. import engine as E


get_ndim = E.get_ndim
get_shape = E.get_shape
reshape = E.reshape
permute = E.permute
repeat = E.repeat


"""
Expand a dimension.

    variable, axis -> variable
"""
expand_dims = E.expand_dims


"""
Squeeze a dimension.

    variable, axis -> variable
"""
squeeze = E.squeeze


"""
Concatenate variables.

    variables, axis -> variable
"""
concat = E.concat
stack = E.stack


size = E.size
clone = E.clone
tensor = E.tensor
constant = E.constant
variable = E.variable
constant_or_variable = E.constant_or_variable
to_numpy = E.to_numpy
to_scalar = E.to_scalar
autograd_record = E.autograd_record
backward = E.backward

"""
Get the data of a variable.

    variable -> tensor
"""
data = E.data

"""
Get the gradient of a variable.

    variable -> tensor
"""
gradient = E.gradient

"""
Update a variable, zeroing its gradient.

    variable, tensor ->
"""
update = E.update
