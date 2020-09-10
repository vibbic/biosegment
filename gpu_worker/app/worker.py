import os
from pathlib import Path

from app.celery_app import celery_app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

try:  
   ROOT_DATA_FOLDER = Path(os.environ["ROOT_DATA_FOLDER"])
except KeyError: 
   import sys
   print("Please set the environment variable ROOT_DATA_FOLDER in .env")
   sys.exit(1)

def create_meta(current, total):
    return {
        "current": current,
        "total": total
    }

@celery_app.task(bind=True, acks_late=True)
def test_pytorch(
    self,
    word: str,
) -> str:
    import torch
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    is_available = torch.cuda.is_available()
    word = f"{current_device} {device_count} {device_name} {is_available}"
    return f"test task return {word}"

@celery_app.task(bind=True, acks_late=True)
def train_unet2d(
    self,
    data_dir,
    log_dir,
    retrain_model = None,
    seed=0,
    device=0,
    print_stats=50,
    input_size=(256,256),
    fm=1,
    levels=4,
    dropout=0.0,
    norm="instance",
    activation="relu",
    in_channels=1,
    classes_of_interest=(0,1,2),
    orientations=(0,),
    loss="ce",
    lr=1e-3,
    step_size=10,
    gamma=0.9,
    epochs=50,
    len_epoch=100,
    test_freq=1,
    train_batch_size=1,
    test_batch_size=1,
    **kwargs,
) -> str:
    import os

    import torch
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
    train = StronglyLabeledVolumeDataset(os.path.join(data_dir, 'train'),
                                        os.path.join(data_dir, 'train_labels'),
                                        input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
                                        in_channels=in_channels, batch_size=train_batch_size,
                                        orientations=orientations)
    test = StronglyLabeledVolumeDataset(os.path.join(data_dir, 'test'),
                                        os.path.join(data_dir, 'test_labels'),
                                        input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
                                        in_channels=in_channels, batch_size=test_batch_size,
                                        orientations=orientations)
    train_loader = DataLoader(train, batch_size=train_batch_size)
    test_loader = DataLoader(test, batch_size=test_batch_size)

    """
        Build the network
    """
    print_frm('Building the network')
    if retrain_model:
        net = torch.load(retrain_model)
    else:
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
    # TODO use on_success syntax
    # if metadata for segmentation creation is present
    if len(kwargs) > 0:
        logger.info(f"Running subtask with kwargs {kwargs}")
        task = celery_app.send_task(
            "app.worker.create_model_from_retraining", 
            # TODO better routing of tasks
            queue="main-queue",
            kwargs={
            "obj_in": kwargs["obj_in"],
            "owner_id": kwargs["owner_id"],
            "project_id": kwargs["project_id"],
        })
        logger.info(f"Subtask {task}")
    else:
        logger.info(f"Not subtask with kwargs {kwargs}")



@celery_app.task(bind=True, acks_late=True)
def infer_unet2d(
    self,
    data_dir,
    model,
    write_dir,
    input_size=(256,256), 
    in_channels=1,
    test_batch_size=1,
    orientations=[0],
    len_epoch=100,
    classes_of_interest=(0, 1, 2),
    **kwargs,
):
    import os
    from pathlib import Path
    import torch
    import numpy as np

    from neuralnets.util.io import print_frm, read_png, write_volume
    from neuralnets.util.validation import segment
    from neuralnets.data.datasets import StronglyLabeledVolumeDataset, UnlabeledVolumeDataset

    def infer(net, data, input_size, in_channels=1, batch_size=1, write_dir=None,
             val_file=None, writer=None, epoch=0, track_progress=False, device=0, orientations=(0,), normalization='unit'):
        # compute segmentation for each orientation and average results
        segmentation = np.zeros((net.out_channels, *data.shape))
        progress=3
        for orientation in orientations:
            segmentation += segment(data, net, input_size, train=False, in_channels=in_channels, batch_size=batch_size,
                                    track_progress=track_progress, device=device, orientation=orientation, normalization=normalization)
            progress += 1
            self.update_state(state="PROGRESS", meta=create_meta(progress, 10))
        segmentation = segmentation / len(orientations)
        return segmentation

    def write_out(write_dir, segmentation, threshold=0.2, classes_of_interest=(0, 1, 2), type='pngseq'):
        # (3, 64, 512, 512)
        # print(segmentation.shape)
        image_array = np.zeros(segmentation.shape[1:])
        maximums = np.argmax(segmentation, axis=0)
        # print(segmentation[0].shape)
        # print(maximums.shape)
        # print(np.max(maximums))
        # print(np.min(maximums))
        # image_array = np.zeros((256, 256))
        for i in range(1, len(classes_of_interest)):
            is_maximum_for_interest = maximums == i
            image_array[is_maximum_for_interest] = i
        write_volume(image_array, write_dir, type=type, index_inc=1)

    model = ROOT_DATA_FOLDER / model
    data_dir = ROOT_DATA_FOLDER / data_dir
    write_dir = ROOT_DATA_FOLDER / write_dir
        
    logger.info(f"model {model}")
    assert model.is_file()
    logger.info(f"data_dir {data_dir}")
    assert data_dir.is_dir()
    logger.info(f"write_dir {write_dir}")
    try:
        write_dir.mkdir(parents=True)
    except FileExistsError:
         logger.error("Write dir already exists: {write_dir}")

    input_shape = (1, input_size[0], input_size[1])

    print_frm('Reading dataset')
    # TODO support multiple dataset classes (labeled, unlabeled)
    # test = UnlabeledVolumeDataset(data_dir,
    #                                 input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
    #                                 in_channels=in_channels, batch_size=test_batch_size,
    #                                 orientations=orientations)

    test = UnlabeledVolumeDataset(data_dir,
                                    input_shape=input_shape, len_epoch=len_epoch, type='pngseq',
                                    in_channels=in_channels, batch_size=test_batch_size,
                                    orientations=orientations)

    print_frm('Loading model')
    net = torch.load(model)
    print_frm('Segmenting')
    self.update_state(state="PROGRESS", meta=create_meta(1, 10))
    segmentation = infer(net, test.data, orientations=orientations, input_size=input_size, in_channels=in_channels)

    if write_dir:
        self.update_state(state="PROGRESS", meta=create_meta(9, 10))
        # TODO also use Paths in neuralnets
        write_out(str(write_dir), segmentation, classes_of_interest=classes_of_interest)
    
    # TODO use on_success syntax
    # if metadata for segmentation creation is present
    if len(kwargs) > 0:
        logger.info(f"Running subtask with kwargs {kwargs}")
        task = celery_app.send_task(
            "app.worker.create_segmentation_from_inference", 
            # TODO better routing of tasks
            queue="main-queue",
            kwargs={
            "obj_in": kwargs["obj_in"],
            "owner_id": kwargs["owner_id"],
            "dataset_id": kwargs["dataset_id"],
            "model_id": kwargs["model_id"],
        })
        logger.info(f"Subtask {task}")
    else:
        logger.info(f"Not subtask with kwargs {kwargs}")