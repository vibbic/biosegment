from raven import Client

from app.core.celery_app import celery_app
from app.core.config import settings

client_sentry = Client(settings.SENTRY_DSN)


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    return f"test task return {word}"

@celery_app.task(acks_late=True)
def test_pytorch(word: str) -> str:
    import torch
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    is_available = torch.cuda.is_available()
    word = f"{current_device} {device_count} {device_name} {is_available}"
    return f"test task return {word}"

def infer(
    model="unet_2d/best_checkpoint.pytorch", 
    data_dir="/data/EMBL/test", 
    input_size="256,256", 
    in_channels=1,
    test_batch_size=1,
    orientations="0",
    len_epoch=100
):
    import os

    import torch
    import numpy as np

    from app.neuralnets.util.io import print_frm, read_png, write_volume
    from app.neuralnets.util.validation import segment
    from app.neuralnets.data.datasets import StronglyLabeledVolumeDataset, UnlabeledVolumeDataset

    from app.neuralnets.inference.inference import infer, write_out

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