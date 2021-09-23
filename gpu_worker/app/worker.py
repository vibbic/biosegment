import os
from pathlib import Path

from app.celery_app import celery_app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

try:
   # expect to be in gpu_worker/ and data/ is located at ..
   os.chdir(Path(".."))
   # but allow for absolute ROOT_DATA_FOLDER
   ROOT_DATA_FOLDER = Path(os.environ["ROOT_DATA_FOLDER"]).resolve()
   print(f"Root data folder {ROOT_DATA_FOLDER}")
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
    annotation_dir,
    resolution,
    classes_of_interest,
    retrain_model = None,
    # note: duplication with backend schema TrainingTaskBase
    seed=0,
    device=0,
    print_stats=50,
    fm=16,
    levels=4,
    dropout=0.0,
    norm="instance",
    activation="relu",
    in_channels=1,
    orientations='z',
    loss="ce",
    lr=1e-3,
    step_size=10,
    gamma=0.9,
    epochs=50,
    len_epoch=100,
    test_freq=1,
    train_batch_size=4,
    test_batch_size=4,
    test_size=0.33,
    **kwargs,
) -> str:
    import json

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.loggers import TensorBoardLogger

    from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
    from neuralnets.networks.unet import UNet2D
    from neuralnets.util.augmentation import Compose, Rotate90, Flip, ContrastAdjust, CleanDeformedLabels, RandomDeformation, AddNoise
    from neuralnets.util.tools import set_seed
    from neuralnets.util.io import read_volume

    from app.shape_utils import annotations_to_png

    self.update_state(state="PROGRESS", meta=create_meta(1, 10))

    """
    Fix seed (for reproducibility)
    """
    set_seed(seed)

    """
        Setup logging directory
    """
    log_dir = (ROOT_DATA_FOLDER / log_dir).parent
    data_dir = ROOT_DATA_FOLDER / data_dir
    annotations_dir_json = ROOT_DATA_FOLDER / annotation_dir

    logger.info('Setting up log directories')
    log_dir.mkdir(parents=True, exist_ok=True)

    """
        Convert JSON annotations to .pngs
    """
    with open(annotations_dir_json, 'r') as fp:
        annotations_data = json.load(fp)
    annotations_dir_png = log_dir / "annotations"

    try:
        # TODO define behaviour if folder exists
        annotations_dir_png.mkdir(parents=True, exist_ok=True)
        logger.info(annotations_data)
        # TODO get max slice from dataset dimensions, support for 3D
        logger.info(resolution)
        for slice_id in range(resolution["z"]):
            slice_id = str(slice_id)
            if slice_id in annotations_data:
                annotations = annotations_data[slice_id]
            else:
                annotations = None
            annotations_to_png(
                # TODO use dataset dimensions
                width=resolution["x"],
                height=resolution["y"],
                annotations=annotations,
                write_to=str(annotations_dir_png / f"{int(slice_id):04d}.png"),
            )
    except Exception as e:
        logger.error(f"Error converting annotations: {e}")
        return None
    logger.info(f"data dir: {data_dir}")
    logger.info(f"pngs folder: {annotations_dir_png}")
    annotations = read_volume(str(annotations_dir_png), type='pngseq')

    """
        Load the data
    """
    # TODO don't hard code these
    input_shape = (1, 256, 256)
    num_workers = 12
    type = 'pngseq'
    split = [.50, .75]
    num_layers = 4
    gpus = [0]
    accelerator = 'dp'
    log_freq = 50
    log_refresh_rate = -1
    k = 16
    bn_size = 2
    orientations = 'z'

    logger.info('Loading data')
    transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), ContrastAdjust(adj=0.1),
                        RandomDeformation(), AddNoise(sigma_max=0.05), CleanDeformedLabels(classes_of_interest)])
    train = LabeledVolumeDataset(str(data_dir), annotations, input_shape=input_shape, type=type,
                                batch_size=train_batch_size, transform=transform, range_split=(0, split[0]),
                                range_dir=orientations)
    val = LabeledSlidingWindowDataset(str(data_dir), annotations, input_shape=input_shape, type=type,
                                    batch_size=test_batch_size, range_split=(split[0], split[1]),
                                    range_dir=orientations)
    test = LabeledSlidingWindowDataset(str(data_dir), annotations, input_shape=input_shape, type=type,
                                    batch_size=test_batch_size, range_split=(split[1], 1),
                                    range_dir=orientations)
    train_loader = DataLoader(train, batch_size=train_batch_size, num_workers=num_workers,
                            pin_memory=True)
    val_loader = DataLoader(val, batch_size=test_batch_size, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test, batch_size=test_batch_size, num_workers=num_workers,
                            pin_memory=True)
    logger.info('Label distribution: ')
    for i in range(len(classes_of_interest)):
        logger.info('    - Class %d: %.3f (train) - %.3f (val) - %.3f (test)' %
                (train.label_stats[0][i][0], train.label_stats[0][i][1],
                val.label_stats[0][i][1], test.label_stats[0][i][1]))
    logger.info('    - Unlabeled pixels: %.3f (train) - %.3f (val) - %.3f (test)' %
            (train.label_stats[0][-1][1], val.label_stats[0][-1][1], test.label_stats[0][-1][1]))

    """
        Build the network
    """
    logger.info('Building the network')
    net = UNet2D(feature_maps=fm, levels=levels, dropout_enc=dropout,
                  dropout_dec=dropout, norm=norm, activation=activation,
                  coi=classes_of_interest, 
                #   num_layers=num_layers, k=k, bn_size=bn_size,
                  loss_fn=loss)

    # if retrain_model:
    #     net.load_state_dict(torch.load(ROOT_DATA_FOLDER / retrain_model))

    """
    Train the network
    """
    logger.info('Starting training')
    logger.info('Training with loss: %s' % loss)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    # logger = TensorBoardLogger(log_dir, name="my_model_corrected")
    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, accelerator=accelerator,
                        default_root_dir=log_dir, flush_logs_every_n_steps=log_freq,
                        log_every_n_steps=log_freq, callbacks=[lr_monitor])
    # TODO overwrite progress bar to do update_state
    self.update_state(state="PROGRESS", meta=create_meta(2, 10))
    trainer.fit(net, train_loader, val_loader)
    self.update_state(state="PROGRESS", meta=create_meta(9, 10))

    """
    Testing the network
    """
    logger.info('Testing network')
    trainer.test(net, test_loader)

    """
    Saving the new model
    """
    logger.info('Saving model')
    location = ROOT_DATA_FOLDER / kwargs["obj_in"]['location']
    torch.save(net.state_dict(), str(location))
    
    # TODO support ONNX model export 
    # net.to_onnx(location.parent / location.name + ".onxx", input_sample=train[0][0], export_params=True)

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
def infer_unet2d(  # TODO: rename to just "infer", also 2.5d and 3d nets are supported here
    self,
    data_dir,
    model,
    write_dir,
    file_type,
    input_size=(256,256), 
    in_channels=1,
    orientations=(0,),
    classes_of_interest=(0, 1, 2),
    **kwargs,
):
    import numpy as np
    from app.file_types import FileType

    from neuralnets.util.tools import load_net
    from neuralnets.util.io import write_volume
    from neuralnets.util.validation import segment
    from neuralnets.data.datasets import SlidingWindowDataset

    def write_out(write_dir, segmentation, classes_of_interest, file_type):
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
        write_volume(image_array, write_dir, type=file_type, index_inc=1)

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
         return

    input_shape = (1, input_size[0], input_size[1])

    file_type = FileType(file_type)
    if not file_type.is_dir():
        try:
            # use first file in raw/ as data 
            data_dir = list(data_dir.iterdir())[0]
            # give segmentation same name
            write_dir = write_dir / f"segmentation_{data_dir.name}"
        except Exception as e:
            logger.error(f"Could not find data for data_dir={data_dir} and file_type={file_type}")
            return

    logger.info('Reading dataset')
    self.update_state(state="PROGRESS", meta=create_meta(1, 10))
    data = SlidingWindowDataset(str(data_dir), input_shape=input_shape, type=file_type.value, in_channels=in_channels)

    logger.info('Loading model')
    net = load_net(model)
    logger.info('Segmenting')
    self.update_state(state="PROGRESS", meta=create_meta(2, 10))
    segmentation = segment(data.data[0], net, input_size, train=False, in_channels=in_channels, device=0, orientations=orientations)

    self.update_state(state="PROGRESS", meta=create_meta(9, 10))
    if write_dir:
        # TODO also use Paths in neuralnets
        write_out(str(write_dir), segmentation, file_type=file_type.value, classes_of_interest=classes_of_interest)
    
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