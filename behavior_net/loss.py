import torch
import torch.nn as nn
import road_matching

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class UncertaintyRegressionLoss(nn.Module):
    """
    Regression with uncertainty estimation (with element-wise weight map)
    choice == 'mse' for L2 loss or 'mae' for L1 loss
    """

    def __init__(self, choice='mse'):
        super(UncertaintyRegressionLoss, self).__init__()
        self.choice = choice

    def __call__(self, y_pred_mean, y_pred_std, y_true, weight=None):

        if self.choice == 'mse':
            diff_map_mean = (y_pred_mean - y_true)**2
            diff_map_std = (torch.abs(y_pred_mean - y_true) - y_pred_std) ** 2
            diff_map = diff_map_mean + diff_map_std
        elif self.choice == 'mae':
            diff_map_mean = torch.abs(y_pred_mean - y_true)
            diff_map_std = torch.abs(torch.abs(y_pred_mean - y_true) - y_pred_std)
            diff_map = diff_map_mean + diff_map_std
        elif self.choice == 'cos_sin_heading_mae':
            diff_map = torch.abs(y_pred_mean - y_true)
        else:
            raise NotImplementedError(
                'Unknown loss function type %s ' % self.choice)

        if weight is not None:
            loss = torch.sum(diff_map*weight) / (torch.sum(weight) + 1e-9)
        else:
            loss = torch.mean(diff_map)

        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

