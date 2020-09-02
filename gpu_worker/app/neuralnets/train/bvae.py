"""
    This is a script that illustrates training a beta variational autoencoder
"""

"""
    Necessary libraries
"""
import argparse
import os

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from neuralnets.data.datasets import UnlabeledVolumeDataset
from neuralnets.networks.bvae import BVAE
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.losses import get_loss_function
from neuralnets.util.tools import set_seed

"""
    Parse all the arguments
"""
print_frm('Parsing arguments')
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--seed", help="Seed for randomization", type=int, default=0)
parser.add_argument("--device", help="GPU device for computations", type=int, default=0)
parser.add_argument("--log_dir", help="Logging directory", type=str, default="bvae")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses",
                    type=int, default=50)

# network parameters
parser.add_argument("--data_dir", help="Data directory", type=str, default="../../../data")
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="64,64")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=128)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)",
                    type=int, default=4)
parser.add_argument("--dropout", help="Dropout", type=float, default=0.0)
parser.add_argument("--norm", help="Normalization in the network (batch or instance)", type=str, default="batch")
parser.add_argument("--activation", help="Non-linear activations in the network", type=str, default="relu")
parser.add_argument("--beta", help="Beta value for the KL loss", type=float, default=0.0)
parser.add_argument("--bottleneck", help="Dimensionality of the bottleneck", type=int, default=128)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay",
                    type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=1)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=200)
parser.add_argument("--len_epoch", help="Number of iteration in each epoch", type=int, default=100)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=4)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=4)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
loss_rec_fn = get_loss_function('mse')
loss_kl_fn = get_loss_function('kld')

"""
Fix seed (for reproducibility)
"""
set_seed(args.seed)

"""
    Setup logging directory
"""
print_frm('Setting up log directories')
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

"""
    Load the data
"""
input_shape = (1, args.input_size[0], args.input_size[1])
print_frm('Loading data')
augmenter = Compose([ToFloatTensor(device=args.device), Rotate90(), FlipX(prob=0.5), FlipY(prob=0.5),
                     ContrastAdjust(adj=0.1),
                     RandomDeformation_2D(input_shape[1:], grid_size=(64, 64), sigma=0.01, device=args.device)])
train = UnlabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EPFL/train'), input_shape=input_shape,
                               len_epoch=args.len_epoch, type='pngseq')
test = UnlabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EPFL/test'), input_shape=input_shape,
                              len_epoch=args.len_epoch, type='pngseq')
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.train_batch_size)

"""
    Build the network
"""
print_frm('Building the network')
net = BVAE(beta=args.beta, input_size=args.input_size, bottleneck_dim=args.bottleneck, feature_maps=args.fm,
           levels=args.levels, dropout_enc=args.dropout, dropout_dec=args.dropout, norm=args.norm,
           activation=args.activation)

"""
    Setup optimization for training
"""
print_frm('Setting up optimization for training')
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the network
"""
print_frm('Starting training')
net.train_net(train_loader, test_loader, loss_rec_fn, loss_kl_fn, optimizer, args.epochs, scheduler=scheduler,
              augmenter=augmenter, print_stats=args.print_stats, log_dir=args.log_dir, device=args.device)

print_frm('Finished!')
