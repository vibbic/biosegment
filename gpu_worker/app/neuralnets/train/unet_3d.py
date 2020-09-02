"""
    This is a script that illustrates training a 3D U-Net
"""

"""
    Necessary libraries
"""
import argparse
import os

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from neuralnets.data.datasets import StronglyLabeledVolumeDataset
from neuralnets.networks.unet import UNet3D
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.losses import get_loss_function
from neuralnets.util.tools import set_seed
from neuralnets.util.validation import validate

"""
    Parse all the arguments
"""
print_frm('Parsing arguments')
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--seed", help="Seed for randomization", type=int, default=0)
parser.add_argument("--device", help="GPU device for computations", type=int, default=0)
parser.add_argument("--log_dir", help="Logging directory", type=str, default="unet_3d")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses",
                    type=int, default=50)

# network parameters
parser.add_argument("--data_dir", help="Data directory", type=str, default="../../../data")
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="16,128,128")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=16)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)",
                    type=int, default=4)
parser.add_argument("--dropout", help="Dropout", type=float, default=0.0)
parser.add_argument("--norm", help="Normalization in the network (batch or instance)", type=str, default="instance")
parser.add_argument("--activation", help="Non-linear activations in the network", type=str, default="relu")
parser.add_argument("--classes_of_interest", help="List of indices that correspond to the classes of interest",
                    type=str, default="0,1")
parser.add_argument("--orientations", help="Orientations to consider for training", type=str, default="0,1,2")

# optimization parameters
parser.add_argument("--loss", help="Specifies the loss function (and optionally, additional parameters separated by "
                                   "hashtags and colons that specify the name) used for optimization",
                    type=str, default="ce")
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay",
                    type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=200)
parser.add_argument("--len_epoch", help="Number of iteration in each epoch", type=int, default=100)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=1)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
args.classes_of_interest = [int(c) for c in args.classes_of_interest.split(',')]
args.orientations = [int(c) for c in args.orientations.split(',')]
loss_fn = get_loss_function(args.loss)

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
print_frm('Loading data')
augmenter = Compose([ToFloatTensor(device=args.device), Rotate90(), FlipX(prob=0.5), FlipY(prob=0.5),
                     ContrastAdjust(adj=0.1, include_segmentation=True),
                     RandomDeformation_3D(args.input_size[1:], grid_size=(64, 64), sigma=0.01, device=args.device,
                                          include_segmentation=True),
                     AddNoise(sigma_max=0.05, include_segmentation=True)])
train = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EPFL/train'),
                                     os.path.join(args.data_dir, 'EM/EPFL/train_labels'),
                                     input_shape=args.input_size, len_epoch=args.len_epoch, type='pngseq',
                                     orientations=args.orientations)
test = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EPFL/test'),
                                    os.path.join(args.data_dir, 'EM/EPFL/test_labels'),
                                    input_shape=args.input_size, len_epoch=args.len_epoch, type='pngseq',
                                    orientations=args.orientations)
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.train_batch_size)

"""
    Build the network
"""
print_frm('Building the network')
net = UNet3D(feature_maps=args.fm, levels=args.levels, dropout_enc=args.dropout, dropout_dec=args.dropout,
             norm=args.norm, activation=args.activation, coi=args.classes_of_interest)

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
net.train_net(train_loader, test_loader, loss_fn, optimizer, args.epochs, scheduler=scheduler,
              augmenter=augmenter, print_stats=args.print_stats, log_dir=args.log_dir, device=args.device)

"""
    Validate the trained network
"""
validate(net, test.data, test.labels, args.input_size, batch_size=args.test_batch_size,
         write_dir=os.path.join(args.log_dir, 'segmentation_final'), classes_of_interest=args.classes_of_interest,
         val_file=os.path.join(args.log_dir, 'validation_final.npy'))
net = torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch'))
validate(net, test.data, test.labels, args.input_size, batch_size=args.test_batch_size,
         write_dir=os.path.join(args.log_dir, 'segmentation_best'), classes_of_interest=args.classes_of_interest,
         val_file=os.path.join(args.log_dir, 'validation_best.npy'))

print_frm('Finished!')
