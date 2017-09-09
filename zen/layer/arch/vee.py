from types import ModuleType


"""
Magic non-zero value returned for every interior > comparison in sequence
creation.  This can be checked for to tell if the user forgot to terminate a >
sequence with a `Z` (which is necessary so the > cache can know when to
pop its internal stack).
"""
_UNTERMINATED_VEE_SEQUENCE = object()


def unterminated_vee_sequence():
    return _UNTERMINATED_VEE_SEQUENCE


class VeeCache(object):
    """
    Cache of > sequences currently being created.
    """

    def __init__(self):
        self.next_gt_id = 0
        self.seq_stack = []

    def extract_sequence(self):
        from .sequence import Sequence
        specs = self.seq_stack.pop()
        for spec in specs[1:]:
            del spec.gt_id
        return Sequence(*specs)

    def add_pair(self, left, right):
        # The special sequence termination node goes at the end.
        assert left is not Z

        # Right is newly seen and is assigned a new gt_id.
        right.gt_id = self.next_gt_id
        self.next_gt_id += 1

        # If it's the beginning of a subsequence, push to stack.
        if not hasattr(left, 'gt_id'):
            self.seq_stack.append([])
            self.seq_stack[-1].append(left)

        # If we're the termination node, extract the sequence.
        if right is Z:
            return self.extract_sequence()

        # It was a normal node, so keep going.
        self.seq_stack[-1].append(right)

        # Return non-False to prevent short-circuit > evaluation.
        return unterminated_vee_sequence()


_VEE_CACHE = VeeCache()


class Vee(object):
    """
    Sequences of these can be created using the > ("vee") operator.
    """

    def __gt__(self, right):
        if isinstance(right, ModuleType):
            right = Z
        return _VEE_CACHE.add_pair(self, right)


"""
The > sequence terminator.
"""
Z = Vee()
