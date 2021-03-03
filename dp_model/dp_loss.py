import torch.nn as nn
import torch
def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y2 = y + 1e-16
    n = y.shape[0]
    loss = loss_func(x, y2) / n
    #print(loss)
    return loss

#def js(x,y):
#    mean_div = (torch.exp(x) + y ) / 2
#    return nn.KLDivLoss(reduction='sum')(x,mean_div) / 2 + nn.KLDivLoss(reduction='sum')(np.log(y + 1e-16) + 1e-16,mean_div) / 2
def my_HistLoss(x ,y):
    n = y.shape[0]
    res = 0
    for batch in range(n):
        sim = torch.sum( torch.pow((x[batch,...] - y[batch,...]), 2) / (x[batch,...] + y[batch,...]) ) / 2
        res += sim
    return res / n

def my_JSLoss(x, y):
    loss_func = nn.KLDivLoss(reduction='sum')
    eps = 1e-16
    n = y.shape[0]
    mean_div = (torch.exp(x) + y ) / 2 + eps

    loss_1 = loss_func(x + eps,mean_div) / n
    loss_2 = loss_func(torch.log(y + eps) + eps,mean_div) / n
    return loss_1 / 2. + loss_2 / 2.

def my_CDFHistLoss(x, y):
    batch = y.shape[0]
    for batch_ in range(batch):
        pass

def my_HistSimLoss(x, y):
    n = y.shape[0]
    res = 0
    for batch_ in range(n):
        sim = torch.sum(1 - torch.abs(x[batch_,...] - y[batch_,...]) / torch.max(x[batch_,...], y[batch_,...]), dim =1)
        res+= sim
    