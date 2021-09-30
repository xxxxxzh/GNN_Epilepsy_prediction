import torch
import torch.nn as nn
def binary_cross_entropy(pred,y):
    # print(pred.device,y.device)
    return nn.BCELoss()(pred,y)