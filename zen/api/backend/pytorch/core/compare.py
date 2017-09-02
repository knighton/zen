import torch


def equal(x, y):
    return x == y


def greater_equal(x, y):
    return x >= y


def greater(x, y):
    return x > y


def less_equal(x, y):
    return x <= y


def less(x, y):
    return x < y


def maximum(x, y):
    return torch.maximum(x, y)


def minimum(x, y):
    return torch.minimum(x, y)


def not_equal(x, y):
    return x != y
