import os

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar

from neuralnets.util.io import write_volume, print_frm
from neuralnets.util.metrics import jaccard, accuracy_metrics, hausdorff_distance
from neuralnets.util.tools import gaussian_window, tensor_to_device, module_to_device, normalize

def sliding_window_multichannel(image, step_size, window_size, in_channels=1, track_progress=False, normalization='unit'):
    """
    Iterator that acts as a sliding window over a multichannel 3D image

    :param image: multichannel image (4D array)
    :param step_size: step size of the sliding window (3-tuple)
    :param window_size: size of the window (3-tuple)
    :param in_channels: amount of subsequent slices that serve as input for the network (should be odd)
    :param track_progress: optionally, for tracking progress with progress bar
    :param normalization: type of data normalization (unit, z or minmax)
    """

    # adjust z-channels if necessary
    window_size = np.asarray(window_size)
    is2d = window_size[0] == 1
    if is2d:  # 2D
        window_size[0] = in_channels

    # define range
    zrange = [0]
    while zrange[-1] < image.shape[1] - window_size[0]:
        zrange.append(zrange[-1] + step_size[0])
    zrange[-1] = image.shape[1] - window_size[0]
    yrange = [0]
    while yrange[-1] < image.shape[2] - window_size[1]:
        yrange.append(yrange[-1] + step_size[1])
    yrange[-1] = image.shape[2] - window_size[1]
    xrange = [0]
    while xrange[-1] < image.shape[3] - window_size[2]:
        xrange.append(xrange[-1] + step_size[2])
    xrange[-1] = image.shape[3] - window_size[2]

    # loop over the range
    if track_progress:
        bar = Bar('Progress', max=len(zrange) * len(yrange) * len(xrange))
    for z in zrange:
        for y in yrange:
            for x in xrange:

                # yield the current window
                if is2d:
                    input = image[0, z:z + window_size[0], y:y + window_size[1], x:x + window_size[2]]
                else:
                    input = image[:, z:z + window_size[0], y:y + window_size[1], x:x + window_size[2]]
                    yield (z, y, x, image[:, z:z + window_size[0], y:y + window_size[1], x:x + window_size[2]])
                input = normalize(input, type=normalization)
                yield (z, y, x, input)

                if track_progress:
                    bar.next()
    if track_progress:
        bar.finish()


def _init_step_size(step_size, input_shape, is2d):
    if step_size == None:
        if is2d:
            step_size = (1, input_shape[0] // 2, input_shape[1] // 2)
        else:
            step_size = (input_shape[0] // 2, input_shape[1] // 2, input_shape[2] // 2)
    return step_size


def _init_gaussian_window(input_shape, is2d):
    if is2d:
        g_window = gaussian_window((1, input_shape[0], input_shape[1]), sigma=input_shape[-1] / 4)
    else:
        g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)
    return g_window


def _init_sliding_window(data, step_size, input_shape, in_channels, is2d, track_progress, normalization):
    if is2d:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=(1, input_shape[0], input_shape[1]),
                                         in_channels=in_channels, track_progress=track_progress, normalization=normalization)
    else:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=input_shape,
                                         track_progress=track_progress, normalization=normalization)
    return sw


def _orient(data, orientation=0):
    """
    This function essentially places the desired orientation axis to that of the original Z-axis
    For example:
          (C, Z, Y, X) -> (C, Y, Z, X) for orientation=1
          (C, Z, Y, X) -> (C, X, Y, Z) for orientation=2
    Note that applying this function twice corresponds to the identity transform

    :param data: assumed to be of shape (C, Z, Y, X)
    :param orientation: 0, 1 or 2 (respectively for Z, Y or X axis)
    :return: reoriented dataset
    """
    if orientation == 1:
        return np.transpose(data, axes=(0, 2, 1, 3))
    elif orientation == 2:
        return np.transpose(data, axes=(0, 3, 2, 1))
    else:
        return data


def _pad(data, input_shape, in_channels):
    # pad data if input shape is larger than data
    in_shape = input_shape if len(input_shape) == 3 else (1, input_shape[0], input_shape[1])
    pad_width = [[0, 0], None, None, None]
    for d in range(3):
        padding = np.maximum(0, in_shape[d] - data.shape[d + 1])
        before = padding // 2
        after = padding - before
        pad_width[d + 1] = [before, after]

    # pad z-slices if necessary (required if the network uses more than 1 input channel)
    if in_channels > 1:
        c = (in_channels // 2)
        pad_width[1][0] = pad_width[1][0] + c
        pad_width[1][1] = pad_width[1][1] + c

    return np.pad(data, pad_width=pad_width, mode='symmetric'), pad_width


def _crop(data, seg_cum, counts_cum, pad_width):
    return data[:, pad_width[1][0]:data.shape[1] - pad_width[1][1], pad_width[2][0]:data.shape[2] - pad_width[2][1],
           pad_width[3][0]:data.shape[3] - pad_width[3][1]], \
           seg_cum[:, pad_width[1][0]:data.shape[1] - pad_width[1][1], pad_width[2][0]:data.shape[2] - pad_width[2][1],
           pad_width[3][0]:data.shape[3] - pad_width[3][1]], \
           counts_cum[pad_width[1][0]:data.shape[1] - pad_width[1][1], pad_width[2][0]:data.shape[2] - pad_width[2][1],
           pad_width[3][0]:data.shape[3] - pad_width[3][1]]


def _forward_prop(net, x):
    outputs = net(x)
    # if the outputs are a tuple, take the last
    if type(outputs) is tuple:
        outputs = outputs[-1]
    return F.softmax(outputs, dim=1)


def _cumulate_segmentation(seg_cum, counts_cum, outputs, g_window, positions, batch_size, input_shape, in_channels,
                           is2d):
    c = in_channels // 2
    for b in range(batch_size):
        (z_b, y_b, x_b) = positions[b, :]
        # take into account the gaussian filtering
        if is2d:
            z_b += c  # correct channel shift
            seg_cum[:, z_b, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, ...])
            counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
        else:
            seg_cum[:, z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, ...])
            counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window


def _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, in_channels,
                   is2d):
    # convert to tensors and switch to correct device
    inputs = tensor_to_device(torch.FloatTensor(batch), device)
    # forward prop
    outputs = _forward_prop(net, inputs)
    # cumulate segmentation volume
    _cumulate_segmentation(seg_cum, counts_cum, outputs, g_window, positions, batch_size, input_shape, in_channels,
                           is2d)


def segment_multichannel_2d(data, net, input_shape, batch_size=1, step_size=None, train=False, track_progress=False,
                            device=0, normalization='unit'):
    """
    Segment a multichannel 2D image using a specific network

    :param data: 3D array (C, Y, X) representing the multichannel 2D image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (2-tuple)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """

    # make sure we compute everything on the correct device
    module_to_device(net, device)

    # set the network in the correct mode
    if train:
        net.train()
    else:
        net.eval()

    # pad data if necessary
    data, pad_width = _pad(data[:, np.newaxis, ...], input_shape, 1)
    data = data[:, 0, ...]

    # get the amount of channels
    channels = data.shape[0]

    # initialize the step size
    step_size = _init_step_size(step_size, input_shape, True)

    # gaussian window for smooth block merging
    g_window = _init_gaussian_window(input_shape, True)

    # allocate space
    seg_cum = np.zeros((net.out_channels, 1, *data.shape[1:]))
    counts_cum = np.zeros((1, *data.shape[1:]))

    # define sliding window
    sw = _init_sliding_window(data[np.newaxis, ...], [channels, *step_size[1:]], input_shape, channels, True,
                              track_progress, normalization)

    # start prediction
    batch_counter = 0
    batch = np.zeros((batch_size, channels, *input_shape))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:
            # process a single batch
            _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, 1,
                           True)

            # reset batch counter
            batch_counter = 0

    # don't forget to process the last batch
    _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, 1, True)

    # crop out the symmetric extension and compute segmentation

    data, seg_cum, counts_cum = _crop(data[:, np.newaxis, ...], seg_cum, counts_cum, pad_width)
    for c in range(net.out_channels):
        seg_cum[c, ...] = np.divide(seg_cum[c, ...], counts_cum)
    seg_cum = seg_cum[:, 0, ...]

    return seg_cum


def segment_multichannel_3d(data, net, input_shape, in_channels=1, batch_size=1, step_size=None, train=False,
                            track_progress=False, device=0, orientation=0, normalization='unit'):
    """
    Segment a multichannel 3D image using a specific network

    :param data: 4D array (C, Z, Y, X) representing the multichannel 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (either 2 or 3-tuple)
    :param in_channels: amount of subsequent slices that serve as input for the network (should be odd)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientation: orientation to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """

    # make sure we compute everything on the correct device
    module_to_device(net, device)

    # set the network in the correct mode
    if train:
        net.train()
    else:
        net.eval()

    # orient data if necessary
    data = _orient(data, orientation)

    # pad data if necessary
    data, pad_width = _pad(data, input_shape, in_channels)

    # 2D or 3D
    is2d = len(input_shape) == 2

    # get the amount of channels
    channels = data.shape[0]
    if is2d:
        channels = in_channels

    # initialize the step size
    step_size = _init_step_size(step_size, input_shape, is2d)

    # gaussian window for smooth block merging
    g_window = _init_gaussian_window(input_shape, is2d)

    # allocate space
    seg_cum = np.zeros((net.out_channels, *data.shape[1:]))
    counts_cum = np.zeros(data.shape[1:])

    # define sliding window
    sw = _init_sliding_window(data, step_size, input_shape, in_channels, is2d, track_progress, normalization)

    # start prediction
    batch_counter = 0
    batch = np.zeros((batch_size, channels, *input_shape))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:
            # process a single batch
            _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape,
                           in_channels, is2d)

            # reset batch counter
            batch_counter = 0

    # don't forget to process the last batch
    _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, in_channels,
                   is2d)

    # crop out the symmetric extension and compute segmentation
    data, seg_cum, counts_cum = _crop(data, seg_cum, counts_cum, pad_width)
    for c in range(net.out_channels):
        seg_cum[c, ...] = np.divide(seg_cum[c, ...], counts_cum)

    # reorient data to its original orientation
    data = _orient(data, orientation)
    seg_cum = _orient(seg_cum, orientation)
    return seg_cum


def segment_multichannel(data, net, input_shape, in_channels=1, batch_size=1, step_size=None, train=False,
                         track_progress=False, device=0, orientation=0, normalization='unit'):
    """
    Segment a multichannel 2D or 3D image using a specific network

    :param data: 4D array (C, [Z, ]Y, X) representing the multichannel image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (either 2 or 3-tuple)
    :param in_channels: amount of subsequent slices that serve as input for the network (should be odd)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientation: orientation to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """
    if data.ndim == 4:
        return segment_multichannel_3d(data, net, input_shape, in_channels=in_channels, batch_size=batch_size,
                                       step_size=step_size, train=train, track_progress=track_progress, device=device,
                                       orientation=orientation, normalization=normalization)
    else:
        return segment_multichannel_2d(data, net, input_shape, batch_size=batch_size, step_size=step_size, train=train,
                                       track_progress=track_progress, device=device, normalization=normalization)


def segment(data, net, input_shape, in_channels=1, batch_size=1, step_size=None, train=False, track_progress=False,
            device=0, orientation=0, normalization='unit'):
    """
    Segment a 3D image using a specific network

    :param data: 3D array (Z, Y, X) representing the 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of
    :param in_channels: Amount of subsequent slices that serve as input for the network (should be odd)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientation: orientation to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """

    return segment_multichannel(data[np.newaxis, ...], net, input_shape, in_channels=in_channels, batch_size=batch_size,
                                step_size=step_size, train=train, track_progress=track_progress, device=device,
                                orientation=orientation, normalization=normalization)


def validate(net, data, labels, input_size, in_channels=1, classes_of_interest=(0, 1), batch_size=1, write_dir=None,
             val_file=None, writer=None, epoch=0, track_progress=False, device=0, orientations=(0,), normalization='unit'):
    """
    Validate a network on a dataset and its labels

    :param net: image-to-image segmentation network
    :param data: 3D array (Z, Y, X) representing the 3D image
    :param labels: 3D array (Z, Y, X) representing the 3D labels
    :param input_size: size of the inputs (either 2 or 3-tuple) for processing
    :param in_channels: Amount of subsequent slices that serve as input for the network (should be odd)
    :param classes_of_interest: index of the label of interest
    :param batch_size: batch size for processing
    :param write_dir: optionally, specify a directory to write the output
    :param val_file: optionally, specify a file to write the validation results
    :param writer: optionally, summary writer for logging to tensorboard
    :param epoch: optionally, current epoch for logging to tensorboard
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientations: list of orientations to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :return: validation results, i.e. accuracy, precision, recall, f-score, jaccard and dice score
    """

    print_frm('Validating the trained network')

    if write_dir is not None and not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # TODO find solution for circular import between inference and validation on segment
    # compute segmentation for each orientation and average results
    segmentation = np.zeros((net.out_channels, *data.shape))
    for orientation in orientations:
        segmentation += segment(data, net, input_size, train=False, in_channels=in_channels, batch_size=batch_size,
                                track_progress=track_progress, device=device, orientation=orientation, normalization=normalization)
    segmentation = segmentation / len(orientations)

    # compute metrics
    w = labels != 255
    comp_hausdorff = np.sum(labels == 255) == 0
    js = [jaccard(segmentation[i], (labels == classes_of_interest[i]).astype('float'), w=w) for i in
          range(1, len(classes_of_interest))]
    ams = [accuracy_metrics(segmentation[i], (labels == classes_of_interest[i]).astype('float'), w=w) for i in
           range(1, len(classes_of_interest))]
    for i in range(1, len(classes_of_interest)):
        if comp_hausdorff:
            h = hausdorff_distance(segmentation[i], labels)[0]
        else:
            h = -1

        # report results
        print_frm('Validation performance for class %d: ' % (classes_of_interest[i]))
        print_frm('    - Accuracy: %f' % (ams[i - 1][0]))
        print_frm('    - Balanced accuracy: %f' % (ams[i - 1][1]))
        print_frm('    - Precision: %f' % (ams[i - 1][2]))
        print_frm('    - Recall: %f' % (ams[i - 1][3]))
        print_frm('    - F1 score: %f' % (ams[i - 1][4]))
        print_frm('    - Jaccard index: %f' % (js[i - 1]))
        print_frm('    - Hausdorff distance: %f' % (h))

        # write stuff if necessary
        if write_dir is not None:
            print_frm('Writing the output for class %d: ' % (classes_of_interest[i]))
            wdir = os.path.join(write_dir, 'class_' + str(i))
            if not os.path.exists(wdir):
                os.mkdir(wdir)
                write_volume(255 * segmentation[i], wdir, type='pngseq')
        if writer is not None:
            z = data.shape[0] // 2
            N = 1024
            if data.shape[1] > N:
                writer.add_image('val/input', data[z:z + 1, :N, :N], epoch)
                writer.add_image('val/segmentation', segmentation[z:z + 1, :N, :N], epoch)
            else:
                writer.add_image('val/input', data[z:z + 1, ...], epoch)
                writer.add_image('val/segmentation', segmentation[z:z + 1, ...], epoch)
    if val_file is not None:
        np.save(val_file, np.concatenate((np.asarray(js)[:, np.newaxis], np.asarray(ams)), axis=1))
    return js, ams
