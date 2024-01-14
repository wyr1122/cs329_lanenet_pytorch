import torch
import torch.nn as nn
from torch.functional import F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    '''
    Only consider two class now: foreground, background.
    '''

    def __init__(self, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean', device=DEVICE):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001, max=0.999999)
        target_onehot = torch.zeros((target.size(0), self.n_class, target.size(1), target.size(2))).to(self.device)
        loss = 0
        for i in range(self.n_class):
            target_onehot[:, i, ...][target == i] = 1
        for i in range(self.n_class):
            loss -= self.alpha[i] * (1 - pt[:, i, ...]) ** self.gamma * target_onehot[:, i, ...] * torch.log(
                pt[:, i, ...])

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss
