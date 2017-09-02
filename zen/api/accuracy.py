from . import core as C


def binary_accuracy(true, pred):
    ret = C.equal(true, C.round(pred))
    return C.mean(C.cast(ret), -1, False)


def categorical_accuracy(true, pred):
    true = C.argmax(true, -1)
    pred = C.argmax(pred, -1)
    ret = C.equal(true, pred)
    return C.mean(C.cast(ret), -1, False)
