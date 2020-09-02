import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from neuralnets.networks.blocks import Conv2D, Conv3D, Linear


class CNN2D(nn.Module):
    """
    2D classical convolutional neural network (CNN)

    :param optional conv_channels: list of the amount of channels in the convolutional layer
    :param optional fc_channels: list of the amount of channels in the linear layers
    :param optional input_size: size of the inputs (C, Y, X)
    :param optional optional kernel_size: kernel size of the convolutions
    :param optional optional norm: specify normalization ("batch", "instance" or None)
    :param optional optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, conv_channels, fc_channels, input_size, kernel_size=3, norm=None, activation='relu'):
        super(CNN2D, self).__init__()

        self.conv_channels = np.asarray(conv_channels).astype('int')
        self.fc_channels = np.asarray(fc_channels).astype('int')
        self.input_size = np.asarray(input_size).astype('int')
        self.kernel_size = kernel_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_size[0]
        data_size = np.asarray(input_size[1:]).astype('int')
        for i, out_channels in enumerate(conv_channels):
            self.conv_features.add_module('conv%d' % (i + 1),
                                          Conv2D(in_channels, out_channels, kernel_size=kernel_size, norm=norm,
                                                 activation=activation))
            in_channels = out_channels
            data_size = np.divide(data_size, 2).astype('int')

        # full connections
        in_channels = conv_channels[-1] * data_size[0] * data_size[1]
        for i, out_channels in enumerate(fc_channels):
            if i == len(fc_channels) - 1:
                fc = Linear(in_channels, out_channels)
            else:
                fc = Linear(in_channels, out_channels, norm=norm, activation=activation)
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

    def forward(self, inputs):

        outputs = inputs
        for i in range(len(self.conv_channels)):
            outputs = getattr(self.conv_features, 'conv%d' % (i + 1))(outputs)
            outputs = F.max_pool2d(outputs, kernel_size=2)

        outputs = outputs.view(outputs.size(0), -1)
        for i in range(len(self.fc_channels)):
            outputs = getattr(self.fc_features, 'linear%d' % (i + 1))(outputs)

        return outputs


class CNN3D(nn.Module):
    """
    3D classical convolutional neural network (CNN)

    :param optional conv_channels: list of the amount of channels in the convolutional layer
    :param optional fc_channels: list of the amount of channels in the linear layers
    :param optional input_size: size of the inputs (C, Z, Y, X)
    :param optional optional kernel_size: kernel size of the convolutions
    :param optional optional norm: specify normalization ("batch", "instance" or None)
    :param optional optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, conv_channels, fc_channels, input_size, kernel_size=3, norm=None, activation='relu'):
        super(CNN3D, self).__init__()

        self.conv_channels = np.asarray(conv_channels).astype('int')
        self.fc_channels = np.asarray(fc_channels).astype('int')
        self.input_size = np.asarray(input_size).astype('int')
        self.kernel_size = kernel_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_size[0]
        data_size = np.asarray(input_size[1:]).astype('int')
        for i, out_channels in enumerate(conv_channels):
            self.conv_features.add_module('conv%d' % (i + 1),
                                          Conv3D(in_channels, out_channels, kernel_size=kernel_size, norm=norm,
                                                 activation=activation))
            in_channels = out_channels
            data_size = np.divide(data_size, 2).astype('int')

        # full connections
        in_channels = conv_channels[-1] * data_size[0] * data_size[1]
        for i, out_channels in enumerate(fc_channels):
            if i == len(fc_channels) - 1:
                fc = Linear(in_channels, out_channels)
            else:
                fc = Linear(in_channels, out_channels, norm=norm, activation=activation)
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

    def forward(self, inputs):

        outputs = inputs
        for i in range(len(self.conv_channels)):
            outputs = getattr(self.conv_features, 'conv%d' % (i + 1))(outputs)
            outputs = F.max_pool3d(outputs, kernel_size=2)

        outputs = outputs.view(outputs.size(0), -1)
        for i in range(len(self.fc_channels)):
            outputs = getattr(self.fc_features, 'linear%d' % (i + 1))(outputs)

        return outputs
