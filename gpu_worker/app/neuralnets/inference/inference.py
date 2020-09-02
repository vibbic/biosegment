
import os
import argparse

import torch
import numpy as np

from neuralnets.util.io import print_frm, read_png, write_volume
from neuralnets.util.validation import segment
from neuralnets.data.datasets import StronglyLabeledVolumeDataset, UnlabeledVolumeDataset


def infer(net, data, input_size, in_channels=1, batch_size=1, write_dir=None,
             val_file=None, writer=None, epoch=0, track_progress=False, device=0, orientations=(0,), normalization='unit'):
    # compute segmentation for each orientation and average results
    segmentation = np.zeros((net.out_channels, *data.shape))
    for orientation in orientations:
        segmentation += segment(data, net, input_size, train=False, in_channels=in_channels, batch_size=batch_size,
                                track_progress=track_progress, device=device, orientation=orientation, normalization=normalization)
    segmentation = segmentation / len(orientations)
    return segmentation

def write_out(write_dir, segmentation, classes_of_interest=(0, 1), type='pngseq'):
    for i in range(1, len(classes_of_interest)):
        write_volume(255 * segmentation[i], "output", type=type)

# TODO resolve code duplication with validation.py
if __name__ == "__main__":    
    print_frm('Parsing arguments')
    # KEEP IN SYNC WITH MLproject arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Saved neuralnets model", type=str, default="unet_2d/best_checkpoint.pytorch")
    parser.add_argument("--data_dir", help="Input data location", type=str, default="data/EM/EMBL/test")
    parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="256,256")
    parser.add_argument("--in_channels", help="Amount of subsequent slices that serve as input (should be odd)", type=int,
                    default=1)
    parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)
    parser.add_argument("--orientations", help="Orientations to consider for training", type=str, default="0")
    # TODO not needed in UnlabeledVolumeDataset for inferencing
    parser.add_argument("--len_epoch", help="Number of iteration in each epoch", type=int, default=100)

    # New custom args
    parser.add_argument("--write_dir", help="Specify a directory to write the output", type=str, default="")

    args = parser.parse_args()

    args.input_size = [int(item) for item in args.input_size.split(',')]
    args.orientations = [int(c) for c in args.orientations.split(',')]

    input_shape = (1, args.input_size[0], args.input_size[1])

    print_frm('Reading dataset')
    # TODO support multiple dataset classes (labeled, unlabeled)
    test = UnlabeledVolumeDataset(args.data_dir,
                                    input_shape=input_shape, len_epoch=args.len_epoch, type='pngseq',
                                    in_channels=args.in_channels, batch_size=args.test_batch_size,
                                    orientations=args.orientations)

    # test = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EMBL/train'),
    #                                 os.path.join(args.data_dir, 'EM/EMBL/train_labels'),
    #                                 input_shape=input_shape, len_epoch=args.len_epoch, type='pngseq',
    #                                 in_channels=args.in_channels, batch_size=args.test_batch_size,
    #                                 orientations=args.orientations)

    print_frm('Loading model')
    net = torch.load(args.model)
    print_frm('Segmenting')
    segmentation = infer(net, test.data, args.input_size)

    if args.write_dir:
         write_out(args.write_dir, segmentation)
