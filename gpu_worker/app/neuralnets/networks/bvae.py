import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from neuralnets.networks.blocks import UNetConvBlock2D, UNetUpSamplingBlock2D
from neuralnets.util.io import print_frm
from neuralnets.util.tools import module_to_device, tensor_to_device, log_scalars, log_images_2d


def _reparametrise(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())

    return mu + std * eps


class Encoder(nn.Module):
    """
    2D convolutional encoder

    :param optional input_size: size of the inputs that propagate through the encoder
    :param optional bottleneck_dim: dimensionality of the bottleneck
    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, input_size, bottleneck_dim=2, in_channels=1, feature_maps=64, levels=5, norm='instance',
                 dropout=0.0, activation='relu'):
        super(Encoder, self).__init__()

        self.features = nn.Sequential()
        self.input_size = input_size
        self.bottleneck_dim = bottleneck_dim
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        in_features = in_channels
        for i in range(levels):
            out_features = feature_maps // (2 ** i)

            # convolutional block
            conv_block = UNetConvBlock2D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(
            in_features=feature_maps // (2 ** (levels - 1)) * (input_size[0] // 2 ** levels) * (
                    input_size[1] // 2 ** levels), out_features=bottleneck_dim * 2))

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.bottleneck(outputs.view(outputs.size(0), outputs.size(1) * outputs.size(2) * outputs.size(3)))

        return encoder_outputs, outputs


class Decoder(nn.Module):
    """
    2D convolutional decoder

    :param optional input_size: size of the inputs that propagate through the encoder
    :param optional bottleneck_dim: dimensionality of the bottleneck
    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, input_size=512, bottleneck_dim=2, out_channels=2, feature_maps=64, levels=5,
                 norm='instance', dropout=0.0, activation='relu'):
        super(Decoder, self).__init__()

        self.features = nn.Sequential()
        self.input_size = input_size
        self.bottleneck_dim = bottleneck_dim
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        # bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(in_features=bottleneck_dim,
                                                  out_features=feature_maps // (2 ** (levels - 1)) * (
                                                          input_size[0] // 2 ** levels) * (
                                                                       input_size[1] // 2 ** levels)))

        for i in range(levels - 1):
            # upsampling block
            upconv = UNetUpSamplingBlock2D(feature_maps // (2 ** (levels - i - 1)),
                                           feature_maps // (2 ** (levels - i - 2)),
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            conv_block = UNetConvBlock2D(feature_maps // (2 ** (levels - i - 2)), feature_maps // 2 ** (levels - i - 2),
                                         norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # upsampling block
        upconv = UNetUpSamplingBlock2D(feature_maps, feature_maps, deconv=True)
        self.features.add_module('upconv%d' % (levels), upconv)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        encoder_outputs.reverse()

        fm = self.feature_maps // (2 ** (self.levels - 1))
        inputs = self.bottleneck(inputs).view(inputs.size(0), fm, self.input_size[0] // (2 ** self.levels),
                                              self.input_size[1] // (2 ** self.levels))

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            if i < self.levels - 1:
                outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class BVAE(nn.Module):
    """
    2D beta variational autoencoder (VAE)

    :param optional beta: beta value of the autoencoder (beta=1 results in the classical VAE)
    :param optional input_size: size of the inputs that propagate through the encoder
    :param optional bottleneck_dim: dimensionality of the bottleneck
    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional dropout_enc: encoder dropout factor
    :param optional dropout_dec: decoder dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, beta=0, input_size=512, bottleneck_dim=2, in_channels=1, out_channels=1, feature_maps=64,
                 levels=5, norm='instance', activation='relu', dropout_enc=0.0, dropout_dec=0.0):
        super(BVAE, self).__init__()

        self.beta = beta
        self.input_size = input_size
        self.bottleneck_dim = bottleneck_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.encoder_outputs = None
        self.decoder_outputs = None
        self.mu = None
        self.logvar = None
        self.z = None

        # contractive path
        self.encoder = Encoder(input_size=input_size, bottleneck_dim=bottleneck_dim, in_channels=in_channels,
                               feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout_enc,
                               activation=activation)
        # expansive path
        self.decoder = Decoder(input_size=input_size, bottleneck_dim=bottleneck_dim, out_channels=out_channels,
                               feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout_dec,
                               activation=activation)

    def forward(self, inputs):

        # contractive path
        self.encoder_outputs, bottleneck = self.encoder(inputs)

        self.mu = bottleneck[:, :self.bottleneck_dim]
        self.logvar = bottleneck[:, self.bottleneck_dim:]

        # reparameterization
        self.z = _reparametrise(self.mu, self.logvar)

        # expansive path
        self.decoder_outputs, outputs = self.decoder(self.z, self.encoder_outputs)

        return outputs

    def train_epoch(self, loader, loss_rec_fn, loss_kl_fn, optimizer, epoch, augmenter=None, print_stats=1, writer=None,
                    write_images=False, device=0):
        """
        Trains the network for one epoch
        :param loader: dataloader
        :param loss_rec_fn: reconstruction loss function
        :param loss_kl_fn: kullback leibler loss function
        :param optimizer: optimizer for the loss function
        :param epoch: current epoch
        :param augmenter: data augmenter
        :param print_stats: frequency of printing statistics
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average training loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        module_to_device(self, device)
        self.train()

        # keep track of the average losses during the epoch
        loss_rec_cum = 0.0
        loss_kl_cum = 0.0
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # transfer to suitable device
            x = tensor_to_device(data.float(), device)

            # get the inputs and augment if necessary
            if augmenter is not None:
                x = augmenter(x)

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_pred = torch.sigmoid(self(x))

            # compute loss
            loss_rec = loss_rec_fn(x_pred, x)
            loss_kl = loss_kl_fn(self.mu, self.logvar)
            loss = loss_rec + self.beta * loss_kl
            loss_rec_cum += loss_rec.data.cpu().numpy()
            loss_kl_cum += loss_kl.data.cpu().numpy()
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics if necessary
            if i % print_stats == 0:
                print_frm('Epoch %5d - Iteration %5d/%5d - Loss Rec: %.6f - Loss KL: %.6f - Loss: %.6f' % (
                epoch, i, len(loader.dataset) / loader.batch_size, loss_rec, loss_kl, loss))

        # don't forget to compute the average and print it
        loss_rec_avg = loss_rec_cum / cnt
        loss_kl_avg = loss_kl_cum / cnt
        loss_avg = loss_cum / cnt
        print_frm(
            'Epoch %5d - Average train loss rec: %.6f - Average train loss KL: %.6f - Average train loss: %.6f' % (
            epoch, loss_rec_avg, loss_kl_avg, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_rec_avg, loss_kl_avg, loss_avg], ['train/' + s for s in ['loss-rec', 'loss-kl', 'loss']],
                        writer, epoch=epoch)

            # log images if necessary
            if write_images:
                log_images_2d([x, x_pred], ['train/' + s for s in ['x', 'x_pred']], writer, epoch=epoch)

        return loss_avg

    def test_epoch(self, loader, loss_rec_fn, loss_kl_fn, epoch, writer=None, write_images=False, device=0):
        """
        Tests the network for one epoch
        :param loader: dataloader
        :param loss_rec_fn: reconstruction loss function
        :param loss_kl_fn: kullback leibler loss function
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average testing loss over the epoch
        """
        # make sure network is on the gpu and in training mode
        module_to_device(self, device)
        self.eval()

        # keep track of the average losses during the epoch
        loss_rec_cum = 0.0
        loss_kl_cum = 0.0
        loss_cum = 0.0
        cnt = 0

        # start epoch
        z = []
        li = []
        for i, data in enumerate(loader):
            # transfer to suitable device
            x = tensor_to_device(data.float(), device)

            # forward prop
            x_pred = torch.sigmoid(self(x))
            z.append(_reparametrise(self.mu, self.logvar).cpu().data.numpy())
            li.append(x.cpu().data.numpy())

            # compute loss
            loss_rec = loss_rec_fn(x_pred, x)
            loss_kl = loss_kl_fn(self.mu, self.logvar)
            loss = loss_rec + self.beta * loss_kl
            loss_rec_cum += loss_rec.data.cpu().numpy()
            loss_kl_cum += loss_kl.data.cpu().numpy()
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

        # don't forget to compute the average and print it
        loss_rec_avg = loss_rec_cum / cnt
        loss_kl_avg = loss_kl_cum / cnt
        loss_avg = loss_cum / cnt
        print_frm('Epoch %5d - Average test loss rec: %.6f - Average test loss KL: %.6f - Average test loss: %.6f' % (
        epoch, loss_rec_avg, loss_kl_avg, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_rec_avg, loss_kl_avg, loss_avg], ['test/' + s for s in ['loss-rec', 'loss-kl', 'loss']],
                        writer, epoch=epoch)

            # log images if necessary
            if write_images:
                log_images_2d([x, x_pred], ['test/' + s for s in ['x', 'x_pred']], writer, epoch=epoch)

        return loss_avg

    def train_net(self, train_loader, test_loader, loss_rec_fn, loss_kl_fn, optimizer, epochs, scheduler=None,
                  test_freq=1, augmenter=None, print_stats=1, log_dir=None, write_images_freq=1, device=0):
        """
        Trains the network
        :param train_loader: data loader with training data
        :param test_loader: data loader with testing data
        :param loss_rec_fn: reconstruction loss function
        :param loss_kl_fn: kullback leibler loss function
        :param optimizer: optimizer for the loss function
        :param epochs: number of training epochs
        :param scheduler: optional scheduler for learning rate tuning
        :param test_freq: frequency of testing
        :param augmenter: data augmenter
        :param print_stats: frequency of logging statistics
        :param log_dir: logging directory
        :param write_images_freq: frequency of writing images
        :param device: GPU device where the computations should occur
        """
        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print_frm('Epoch %5d/%5d' % (epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_rec_fn=loss_rec_fn, loss_kl_fn=loss_kl_fn, optimizer=optimizer,
                             epoch=epoch, augmenter=augmenter, print_stats=print_stats, writer=writer,
                             write_images=epoch % write_images_freq == 0, device=device)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_rec_fn=loss_rec_fn, loss_kl_fn=loss_kl_fn,
                                            epoch=epoch, writer=writer, write_images=True, device=device)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()
