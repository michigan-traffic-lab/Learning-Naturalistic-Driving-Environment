import torch
import torch.nn as nn


class RegressionAccuracy(nn.Module):
    """
    Standart regression accuracy measurement
    choice == 'r2' for R-square or 'neg_mae' for -1*L1 loss
    """

    def __init__(self, choice='r2'):
        super(RegressionAccuracy, self).__init__()
        self.choice = choice

    def __call__(self, y_pred, y_true, mask=None):

        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        if len(y_true) == 0:
            return 0

        if self.choice == 'r2':
            var_y = torch.var(y_true, unbiased=False)
            acc = 1.0 - torch.mean((y_pred - y_true)**2) / (var_y + 1e-9)
            return acc
        elif self.choice == 'neg_mae':
            acc = -1 * torch.mean(torch.abs(y_pred - y_true))
            return acc
        elif self.choice == 'angle_neg_mae':
            acc = -1 * torch.mean(torch.minimum(torch.abs(y_pred - y_true), 360 - torch.abs(y_pred - y_true)))
            return acc
        else:
            raise NotImplementedError(
                'Unknown acc metric function type %s ' % self.choice)

