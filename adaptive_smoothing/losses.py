import torch
import torch.nn as nn


def reduce(loss, reduction):
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unknown reduction type.")


class DLRLoss(nn.Module):
    def __init__(self, y_target=None, reduction='mean'):
        super().__init__()
        self.y_target = y_target
        self.reduction = reduction

    def forward(self, x, y):
        assert self.y_target is not None
        x_sorted, _ = x.sort(dim=1)
        u = torch.arange(x.shape[0])
        dlr_loss = (-(x[u, y] - x[u, self.y_target]) /
                    (x_sorted[:, -1] - .5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12))
        return reduce(dlr_loss, self.reduction)


class CompLoss(nn.Module):
    def __init__(self, consts=(.5, 1, .5), reduction='mean'):
        super().__init__()
        self.c1, self.c2, self.c3 = consts
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, preds, labels, alphas, sup_labels, scale=1):
        """
            sup_label = 0 if small alpha (use STD network) means lower loss
            sup_label = 1 if large alpha (use ADV network) means lower loss
        """
        if torch.is_tensor(sup_labels) and len(sup_labels.shape) > 0:  # sup_labels is a tensor array
            assert sup_labels.shape == labels.shape
        if torch.is_tensor(scale) and len(scale.shape) > 0:  # scale is a tensor array
            assert scale.shape == labels.shape
        
        ce_loss = self.ce_loss(preds, labels)
        bce_loss = self.bce_loss(alphas, sup_labels)
        comp_loss = self.c1 * bce_loss + self.c2 * ce_loss + self.c3 * bce_loss * ce_loss
        return reduce(comp_loss * scale, self.reduction)


class SimpleCompLoss(nn.Module):
    def __init__(self, base_loss, w=0.00145, reduction='mean'):
        super().__init__()
        self.w = w  # Weight of the alpha loss
        self.base_loss = base_loss  # Base loss must be reduction='none'
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, preds, labels, alphas, sup_labels):
        base_loss = self.base_loss(preds, labels)
        bce_loss = self.bce_loss(alphas, sup_labels)
        sc_loss = (1. - self.w) * base_loss + self.w * bce_loss
        return reduce(sc_loss, self.reduction)
