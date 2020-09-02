from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt
from skimage import measure

from neuralnets.data.datasets import StronglyLabeledStandardDataset, StronglyLabeledVolumeDataset, \
    StronglyLabeledMultiVolumeDataset

from neuralnets.util.tools import tensor_to_device


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss function

    :param initalization weight: weights for the classes (C)
    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :return: focal loss
    """

    def __init__(self, weight=None, device=0):

        super(CrossEntropyLoss, self).__init__()

        self.weight = weight
        self.device = device

        w = None
        if weight is not None:
            w = tensor_to_device(torch.Tensor(weight), device=device)
        self.ce = nn.CrossEntropyLoss(weight=w, reduction="none")

    def forward(self, logits, target, mask=None):

        # compute loss unreduced and reshape to vector
        loss = self.ce(logits, target).view(-1)

        # size averaging if necessary
        if mask is not None:
            loss = loss[mask.view(-1)].mean()
        else:
            loss = loss.mean()

        return loss

    def set_weight(self, weight=None):

        self.weight = weight

        w = None
        if weight is not None:
            w = tensor_to_device(torch.Tensor(weight), device=self.device)

        self.ce = nn.CrossEntropyLoss(weight=w, reduction="none")

    def __str__(self):
        if self.weight is None:
            return "Cross entropy"
        else:
            return "Cross entropy (weights: %s, device: %d)" % (self.weight, self.device)


class FocalLoss(nn.Module):
    """
    Focal loss function (T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar. Focal Loss for Dense Object Detection, 2017)

    :param initalization alpha: weights for the classes (C)
    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :return: focal loss
    """

    def __init__(self, gamma=2, alpha=None):

        super(FocalLoss, self).__init__()

        self.alpha = alpha
        # normalize alpha if necessary
        if self.alpha is not None:
            self.alpha = torch.Tensor(self.alpha / np.sum(self.alpha))
        self.gamma = gamma

    def forward(self, logits, target, mask=None):

        # apply log softmax
        log_p = F.log_softmax(logits, dim=1)
        p = F.softmax(logits, dim=1)

        # channels on the last axis
        input_size = logits.size()
        for d in range(1, len(input_size) - 1):
            log_p = log_p.transpose(d, d + 1)
        log_p = log_p.contiguous()

        # reshape everything
        log_p = log_p[target[:, ...].unsqueeze(-1).repeat_interleave(input_size[1], dim=-1) >= 0]
        log_p = log_p.view(-1, input_size[1])
        p = p.view(-1, input_size[1])
        target = target.view(-1)

        # compute negative log likelihood
        if self.alpha is not None:
            cw = tensor_to_device(self.alpha, device=target.device.index)
        else:
            cw = None
        loss = F.nll_loss((1 - p) ** self.gamma * log_p, target, reduction='none', weight=cw)

        # size averaging if necessary
        if mask is not None:
            loss = loss[mask.view(-1)].mean()
        else:
            loss = loss.mean()

        return loss

    def __str__(self):
        if self.alpha is None:
            return "Focal (alpha: [0.5, 0.5], gamma: %f)"
        else:
            return "Focal (alpha: %s, gamma: %f)" % (self.alpha, self.gamma)


class DiceLoss(nn.Module):
    """
    Dice loss function

    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :return: dice loss
    """

    def forward(self, logits, target, mask=None):

        # precompute for efficiency
        if mask is not None:
            mask = mask.view(-1)

        # apply softmax and compute dice loss for each class
        probs = F.softmax(logits, dim=1)
        dice = 0
        for c in range(1, logits.size(1)):
            p = probs[:, c:c + 1, ...]
            t = (target == c).long()

            # reshape everything to vectors
            p = p.contiguous().view(-1)
            t = t.contiguous().view(-1)

            # mask if necessary
            if mask is not None:
                p = p[mask]
                t = t[mask]

            # dice loss
            numerator = 2 * torch.sum(p * t)
            denominator = torch.sum(p + t)
            dice = dice + (1 - ((numerator + 1) / (denominator + 1)))

        # compute average
        dice = dice / (logits.size(1) - 1)

        return dice

    def __str__(self):
        return "Dice"


class TverskyLoss(nn.Module):
    """
    Tversky loss function (S. S. M. Salehi, D. Erdogmus, and A. Gholipour. Tversky loss function for image segmentation using 3D fully convolutional deep networks, 2017)

    :param initialization c: index of the class of index
    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :return: tversky loss
    """

    def __init__(self, beta=0.5, c=1):
        super(TverskyLoss, self).__init__()

        self.beta = beta
        self.c = c

    def forward(self, logits, target, mask=None):

        # precompute for efficiency
        if mask is not None:
            mask = mask.view(-1)

        # apply softmax and compute dice loss for each class
        probs = F.softmax(logits, dim=1)
        tversky = 0
        for c in range(1, logits.size(1)):
            p = probs[:, c:c + 1, ...]
            t = (target == c).long()

            # reshape everything to vectors
            p = p.contiguous().view(-1)
            t = t.contiguous().view(-1)

            # mask if necessary
            if mask is not None:
                p = p[mask]
                t = t[mask]

            # dice loss
            numerator = torch.sum(p * t)
            denominator = numerator + self.beta * torch.sum((1 - t) * p) + (1 - self.beta) * torch.sum((1 - p) * t)
            tversky = tversky + (1 - ((numerator + 1) / (denominator + 1)))

        # compute average
        tversky = tversky / (logits.size(1) - 1)

        return tversky

    def __str__(self):
        return "Tversky (beta: %f, c: %d)" % (self.beta, self.c)


class LpLoss(nn.Module):
    """
    L_p loss function

    :param initalization p: parameter for the loss function
    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward logits: logits tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: L_p loss
    """

    def __init__(self, p=2, size_average=True):
        super(LpLoss, self).__init__()

        self.p = p
        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.pow(torch.sum(torch.pow(torch.abs(target - target_rec), self.p)), 1 / self.p)
        if self.size_average:
            loss = loss / target.numel()
        return loss

    def __str__(self):
        return "Lp (p: %f, c: %d)" % (self.p)


class L2Loss(nn.Module):
    """
    L_2 loss function

    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward logits: logits tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: L_2 loss
    """

    def __init__(self, size_average=True):
        super(L2Loss, self).__init__()

        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.sqrt(torch.sum(torch.pow(target - target_rec, 2)))
        if self.size_average:
            loss = loss / target.numel()
        return loss

    def __str__(self):
        return "L2"


class MADLoss(nn.Module):
    """
    Mean absolute deviation (MAD) loss function

    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward pred: predictions tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: MAD loss
    """

    def __init__(self, size_average=True):
        super(MADLoss, self).__init__()

        self.size_average = size_average

    def forward(self, pred, target):
        loss = torch.sum(torch.abs(target - pred))
        if self.size_average:
            loss = loss / target.numel()
        return loss

    def __str__(self):
        return "Mean absolute deviation"


class KLDLoss(nn.Module):
    """
    Kullback Leibler divergence (KLD) loss function

    :param forward mu: mean tensor (N_1, N_2, ...)
    :param forward log: logarithmic variance tensor (N_1, N_2, ...)
    :return: KLD loss
    """

    def forward(self, mu, logvar):

        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = 1 + logvar - mu.pow(2) - logvar.exp()
        kld = torch.mean(-0.5 * torch.sum(klds, dim=1), dim=0)

        return kld

    def __str__(self):
        return "Kullback Leibler divergence"


def boundary_weight_map(labels, sigma=20, w0=1):
    """
    Compute the boundary weight map, according to the the original U-Net paper

    :param labels: input tensor
    :param optional sigma: damping parameter
    :param optional w0: initial value of the weight map
    :return boundary weight map as a tensor
    """

    y = labels.cpu().numpy()
    weight = np.ones(y.shape)
    diag = np.sqrt(weight.shape[2] ** 2 + weight.shape[3] ** 2)

    # for each image in the batch
    for b in range(y.shape[0]):

        # compute connected components
        comps = measure.label(y[b, 0, ...])
        n_comps = np.max(comps)

        if n_comps > 0:  # if there is at least one component
            if n_comps > 1:  # if there are at least two components
                # remove every component separately and compute distance transform
                dtfs = np.zeros((n_comps, comps.shape[0], comps.shape[1]))
                for c in range(n_comps):
                    # compute distance transform
                    y_ = copy(y[b, 0, ...])
                    y_[comps == (1 + c)] = 0  # remove component
                    dtfs[c, ...] = distance_transform_edt(1 - y_)
                dtfs_sorted = np.sort(dtfs, axis=0)
                dtf1 = dtfs_sorted[0, ...]
                dtf2 = dtfs_sorted[1, ...]
            else:
                # compute distance transform
                dtf1 = distance_transform_edt(1 - y[b, 0, ...])
                dtf2 = diag
            # update weight map
            weight[b, 0, ...] = weight[b, 0, ...] + w0 * (1 - y[b, 0, ...]) * np.exp(
                - (dtf1 + dtf2) ** 2 / (2 * sigma ** 2))

    return torch.Tensor(weight).cuda()


def _parse_loss_params(t):
    params = {}
    for s in t:
        key, val = s.split(':')
        if val.count(',') > 0:
            val = val.split(',')
            for i in range(len(val)):
                val[i] = float(val[i])
        else:
            val = float(val)
        params[key] = val

    return params


def get_loss_function(s):
    """
    Returns a loss function according to the settings in a string. The string is formatted as follows:
        s = <loss-function-name>[#<param>:<param-value>#<param>:<param-value>#...]
            parameter values are either
                - scalars
                - vectors (written as scalars, separated by commas)

    :param s: loss function specifier string, formatted as shown on top
    :return: the required loss function
    """
    t = s.lower().replace("-", "_").split('#')
    name = t[0]
    params = _parse_loss_params(t[1:])
    if name == "ce" or name == "cross_entropy":
        return CrossEntropyLoss(**params)
    elif name == "fl" or name == "focal":
        return FocalLoss(**params)
    elif name == "dl" or name == "dice":
        return DiceLoss(**params)
    elif name == "tl" or name == "tversky":
        return TverskyLoss(**params)
    elif name == "l1":
        return nn.L1Loss(**params)
    elif name == "mse":
        return nn.MSELoss(**params)
    elif name == "kld":
        return KLDLoss(**params)


def get_balancing_weights(dataset):
    """
    Returns a set of balancing weights for a specific labeled dataset

    :param dataset: labeled dataset instance of
                        - neuralnets.data.datasets.StronglyLabeledStandardDataset
                        - neuralnets.data.datasets.StronglyLabeledVolumeDataset
                        - neuralnets.data.datasets.StronglyLabeledMultiVolumeDataset
    :return: a tuple of balancing weights, if an unsuitable object is provided, it returns None
    """

    if isinstance(dataset, StronglyLabeledStandardDataset) or isinstance(dataset, StronglyLabeledVolumeDataset):
        weight = np.zeros((len(dataset.coi)))
        for i, c in enumerate(dataset.coi):
            weight[i] = 1 / np.count_nonzero(dataset.labels == c)
        weight = weight / np.sum(weight)
        return tuple(weight)
    elif isinstance(dataset, StronglyLabeledMultiVolumeDataset):
        freq = np.zeros((len(dataset.coi)))
        for i, c in enumerate(dataset.coi):
            for labels in dataset.labels:
                freq[i] += np.count_nonzero(labels == c)
        weight = 1 / freq
        weight = weight / np.sum(weight)
        return weight

    return None
