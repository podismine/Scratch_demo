# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
class ADLoss(nn.Module):
    def __init__(self):
        super(ADLoss, self).__init__()
    def forward(self,ins,target):
        #print(target.size()) #N,1
        target =target.view(-1,1)
        batch = target.size(0)
        diff = ins - target
        #print("diff, ",diff.shape)
        res = 0.0
        for i in range(batch):
            for j in range(batch):
                if i == j:
                    continue
                res += torch.pow((diff[i] - diff[j]), 2)
        return res / ( batch * (batch - 1))
class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """
    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        class_weights = self._class_weights(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, _stacklevel=5)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights

def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return dice_sum / class_num


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num


class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):

        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0, 0]).cuda()
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + num_sum[i - 2] + 1) / (torch.sum(num_sum) + 1)
        print(w)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)

        return 1 - 2. * inter / union


def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice


def T(logits, targets):
    return torch.sum(targets[:, 2, :, :, :])


def P(logits, targets):
    return torch.sum(logits[:, 2, :, :, :])


def TP(logits, targets):
    return torch.sum(targets[:, 2, :, :, :] * logits[:, 2, :, :, :])
