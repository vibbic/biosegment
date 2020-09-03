from app.celery_app import celery_app

ROOT_DATA_FOLDER="/home/brombaut/workspace/biosegment/data/"

@celery_app.task(acks_late=True)
def test_pytorch(word: str) -> str:
    import torch
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    is_available = torch.cuda.is_available()
    word = f"{current_device} {device_count} {device_name} {is_available}"
    return f"test task return {word}"

@celery_app.task(acks_late=True)
def train_unet2d(
    seed=0,
    device=0,
    data_dir=f"{ROOT_DATA_FOLDER}",
    log_dir=f"{ROOT_DATA_FOLDER}models/EMBL/test_run1",
    print_stats=50,
    input_size="256,256",
    fm=1,
    levels=4,
    dropout=0.0,
    norm="instance",
    activation="relu",
    in_channels=1,
    classes_of_interest="0,1,2",
    orientations="0",
    loss="ce",
    lr=1e-3,
    step_size=10,
    gamma=0.9,
    epochs=200,
    len_epoch=100,
    test_freq=1,
    train_batch_size=1,
    test_batch_size=1,
) -> str:
    import os

    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose

    from neuralnets.data.datasets import StronglyLabeledVolumeDataset
    from neuralnets.networks.unet import UNet2D
    from neuralnets.util.augmentation import ToFloatTensor, Rotate90, FlipX, FlipY, ContrastAdjust, RandomDeformation_2D, AddNoise
    from neuralnets.util.io import print_frm
    from neuralnets.util.losses import get_loss_function
    from neuralnets.util.tools import set_seed
    from neuralnets.util.validation import validate

    input_size = [int(item) for item in input_size.split(',')]
    classes_of_interest = [int(c) for c in classes_of_interest.split(',')]
    orientations = [int(c) for c in orientations.split(',')]
    loss_fn = get_loss_function(loss)

    """
    Fix seed (for reproducibility)
    """
    set_seed(seed)

    """
        Setup logging directory
    """
    print_frm('Setting up log directories')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    """
        Load the data
    """
    input_shape = (1, input_size[0], input_size[1])
    print_frm('Loading data')
    augmenter = Compose([ToFloatTensor(device=device), Rotate90(), FlipX(prob=0.5), FlipY(prob=0.5),
                        ContrastAdjust(adj=0.1, include_segmentation=True),
                        RandomDeformation_2D(input_shape[1:], grid_size=(64, 64), sigma=0.01, device=device,
                                            include_segmentation=True),
                        AddNoise(sigma_max=0.05, include_segmentation=True)])
    train = StronglyLabeledVolumeDataset(os.path.join(data_dir, 'EM/EMBL/train'),
                                        os.path.join(data_dir, 'EM/EMBL/train_labels'),
                                        input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
                                        in_channels=in_channels, batch_size=train_batch_size,
                                        orientations=orientations)
    test = StronglyLabeledVolumeDataset(os.path.join(data_dir, 'EM/EMBL/test'),
                                        os.path.join(data_dir, 'EM/EMBL/test_labels'),
                                        input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
                                        in_channels=in_channels, batch_size=test_batch_size,
                                        orientations=orientations)
    train_loader = DataLoader(train, batch_size=train_batch_size)
    test_loader = DataLoader(test, batch_size=test_batch_size)

    """
        Build the network
    """
    print_frm('Building the network')
    net = UNet2D(in_channels=in_channels, feature_maps=fm, levels=levels, dropout_enc=dropout,
                dropout_dec=dropout, norm=norm, activation=activation, coi=classes_of_interest)

    """
        Setup optimization for training
    """
    print_frm('Setting up optimization for training')
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    """
        Train the network
    """
    print_frm('Starting training')
    net.train_net(train_loader, test_loader, loss_fn, optimizer, epochs, scheduler=scheduler,
                augmenter=augmenter, print_stats=print_stats, log_dir=log_dir, device=device)



@celery_app.task(acks_late=True)
def infer_unet2d(
    model=f"{ROOT_DATA_FOLDER}models/2d/2d.pytorch", 
    data_dir=f"{ROOT_DATA_FOLDER}EM/EMBL/raw",
    labels_dir=f"{ROOT_DATA_FOLDER}EM/EMBL/labels",
    input_size="256,256", 
    in_channels=1,
    test_batch_size=1,
    orientations="0",
    len_epoch=100,
    write_dir=f"{ROOT_DATA_FOLDER}segmentations/EMBL",
    classes_of_interest=(0, 1, 2)
):
    import os
    import torch
    import numpy as np

    from neuralnets.util.io import print_frm, read_png, write_volume
    from neuralnets.util.validation import segment_multichannel
    from neuralnets.data.datasets import StronglyLabeledVolumeDataset, UnlabeledVolumeDataset

    def infer(net, data, input_size, in_channels=1, batch_size=1, write_dir=None,
             val_file=None, writer=None, epoch=0, track_progress=False, device=0, orientations=(0,), normalization='unit'):
        # compute segmentation for each orientation and average results
        segmentation = np.zeros((net.out_channels, *data.shape))
        for orientation in orientations:
            segmentation += segment_multichannel(data, net, input_size, train=False, in_channels=in_channels, batch_size=batch_size,
                                    track_progress=track_progress, device=device, orientation=orientation, normalization=normalization)
        segmentation = segmentation / len(orientations)
        return segmentation

    def write_out(write_dir, segmentation, threshold=0.5, classes_of_interest=(0, 1, 2), type='pngseq'):
        for i in range(1, len(classes_of_interest)):
            s = segmentation[i]
            above_indices =  s > threshold
            below_indices =  s <= threshold
            s[above_indices] = i
            s[below_indices] = 0
            write_volume(s, write_dir, type=type)

    input_size = [int(item) for item in input_size.split(',')]
    orientations = [int(c) for c in orientations.split(',')]

    input_shape = (2, input_size[0], input_size[1])

    print_frm('Reading dataset')
    # TODO support multiple dataset classes (labeled, unlabeled)
    # test = UnlabeledVolumeDataset(data_dir,
    #                                 input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
    #                                 in_channels=in_channels, batch_size=test_batch_size,
    #                                 orientations=orientations)

    test = StronglyLabeledVolumeDataset(data_dir,
                                    labels_dir,
                                    coi=classes_of_interest,
                                    input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
                                    in_channels=in_channels, batch_size=test_batch_size,
                                    orientations=orientations)

    print_frm('Loading model')
    net = torch.load(model)
    print_frm('Segmenting')
    segmentation = infer(net, test.data, orientations=orientations, input_size=input_size, in_channels=in_channels)

    if write_dir:
         write_out(write_dir, segmentation, classes_of_interest=classes_of_interest)