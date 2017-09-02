import torch


def binary_accuracy(true, pred):
    ret = true == pred.round()
    ret = ret.type(torch.cuda.FloatTensor)
    return 100. * ret.mean()


def categorical_accuracy(true, pred):
    true = true.max(1)[1]
    pred = pred.max(1)[1]
    ret = true == pred
    ret = ret.type(torch.cuda.FloatTensor)
    return 100. * ret.mean()
