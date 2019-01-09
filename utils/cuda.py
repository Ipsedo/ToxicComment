import torch as th


def use_cuda():
    return th.cuda.is_available()
