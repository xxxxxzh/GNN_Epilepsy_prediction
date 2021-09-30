import torch

def accuracy(pred,y, v = 0.5):
    '''

    :param pred:
    :param y:
    :param v:       阈值
    :return:
    '''
    pred[pred >= v] = 1
    pred[pred < v] = 0
    return (pred * y).sum().item() / pred.numel()