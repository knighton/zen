from .. import backend as Z


get_ndim = Z.get_ndim
get_shape = Z.get_shape


"""
Expand a dimension.

    variable, axis -> variable
"""
expand_dims = Z.expand_dims


"""
Squeeze a dimension.

    variable, axis -> variable
"""
squeeze = Z.squeeze


"""
Concatenate variables.

    variables, axis -> variable
"""
concat = Z.concat


count_params = Z.count_params
constant = Z.constant
variable = Z.variable
tensor = Z.tensor
to_numpy = Z.to_numpy
to_scalar = Z.to_scalar
autograd_record = Z.autograd_record

"""
Get the data of a variable.

    variable -> tensor
"""
data = Z.data

"""
Get the gradient of a variable.

    variable -> tensor
"""
gradient = Z.gradient

"""
Update a variable, zeroing its gradient.

    variable, tensor ->
"""
update = Z.update
