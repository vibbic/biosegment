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
def infer(
    model=f"{ROOT_DATA_FOLDER}models/unet_2d/best_checkpoint.pytorch", 
    data_dir=f"{ROOT_DATA_FOLDER}EM/EMBL/raw", 
    input_size="256,256", 
    in_channels=1,
    test_batch_size=1,
    orientations="0",
    len_epoch=100,
    write_dir=f"{ROOT_DATA_FOLDER}segmentations/EMBL"
):
    import os
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
            write_volume(255 * segmentation[i], write_dir, type=type)

    input_size = [int(item) for item in input_size.split(',')]
    orientations = [int(c) for c in orientations.split(',')]

    input_shape = (1, input_size[0], input_size[1])

    print_frm('Reading dataset')
    # TODO support multiple dataset classes (labeled, unlabeled)
    test = UnlabeledVolumeDataset(data_dir,
                                    input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
                                    in_channels=in_channels, batch_size=test_batch_size,
                                    orientations=orientations)

    # test = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EMBL/train'),
    #                                 os.path.join(args.data_dir, 'EM/EMBL/train_labels'),
    #                                 input_shape=input_shape, len_epoch=args.len_epoch, type='pngseq',
    #                                 in_channels=args.in_channels, batch_size=args.test_batch_size,
    #                                 orientations=args.orientations)

    print_frm('Loading model')
    net = torch.load(model)
    print_frm('Segmenting')
    segmentation = infer(net, test.data, input_size)

    if write_dir:
         write_out(write_dir, segmentation)