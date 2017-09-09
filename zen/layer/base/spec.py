from copy import deepcopy


class Spec(object):
    """
    Thing that constructs Operations.

    They live inside Nodes, which provide shapes and dtypes for Operation
    construction from the rest of the neural network.

    Spec vs op:
    * parameter shape and initializer go in the spec
    * the op is initialized with the actual weights as numpy ndarrays
    """
    pass


class TransformSpec(Spec):
    """
    Specification for an Operation (single input, single output layer).
    """

    def build(self, in_shape, in_dtype):
        """
        in_shape, in_dtype -> Transform, out_shape, out_dtype

        Construct the Transform.
        """
        raise NotImplementedError

    def build_multi_input(self, in_shapes, in_dtypes):
        """
        in_shapes, in_dtypes -> Transform, out_shape, out_dtype

        Construct the Transform.  Wraps build().
        """
        if in_shapes is None:
            in_shape = None
        else:
            assert len(in_shapes) == 1
            in_shape = in_shapes[0]
        if in_dtypes is None:
            in_dtype = None
        else:
            assert len(in_dtypes) == 1
            in_dtype = in_dtypes[0]
        return self.build(in_shape, in_dtype)


class MergeSpec(Spec):
    """
    Specification for a Merge (multiple input, single output layer).
    """

    def build_multi_input(self, in_shapes, in_dtypes):
        """
        in_shapes, in_dtypes -> Merge, out_shape, out_dtype

        Construct the Merge.
        """
        raise NotImplementedError
