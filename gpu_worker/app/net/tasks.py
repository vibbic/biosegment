import logging
from app.celery_app import celery_app
from celery.utils.log import get_task_logger
from app.net.NetTask import NetTask
from app.utils import ROOT_DATA_FOLDER, arr_to_str, create_meta, str_to_arr
import numpy as np
from app.file_types import FileType

logger = get_task_logger(__name__)

def hist(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return np.asarray((unique, counts)).T

def prob_to_label(arr, coi):
    image_array = np.zeros(arr.shape[1:], dtype='uint8')
    print(f'image array {image_array.shape}')
    maximums = np.argmax(arr, axis=0)
    max_ind = len(coi)
    if 0 not in coi:
        max_ind += 1
    for i in range(1, max_ind):
        is_maximum_for_interest = maximums == i
        image_array[is_maximum_for_interest] = i
    print(f'image array {image_array.shape}')
    return image_array

@celery_app.task(base=NetTask, acks_late=True)
def infer_region(
    # self,
    region,
    model_path,
    input_size, 
    classes_of_interest,
    **kwargs,
):
    from neuralnets.util.validation import segment
    from neuralnets.util.tools import load_net

    from importlib.metadata import version
    logger.info(version('neuralnets'))

    model = ROOT_DATA_FOLDER / model_path

    logger.info(f"model {model}")
    assert model.is_file()
    # self.set_model_path(model)

    region = str_to_arr(region)
    logger.info(f"region shape {region.shape}")
    logger.info(f"input size {input_size}")
    if list(region.shape) == [1, input_size[0], input_size[1]]:
        region = region[0]
    # assert list(region.shape) == input_size

    m = np.linalg.norm(region)
    region = region / m

    # logger.info('Loading model')
    net = load_net(model)
    # logger.info('Segmenting')
    # logger.info(hist(region)[:10])
    segmentation = segment(region, net, region.shape, train=False, device=0)
    logger.info(f"segmentation shape {segmentation.shape}")
    # logger.info(hist(segmentation)[:10])
    image_array = prob_to_label(segmentation, classes_of_interest)
    logger.info(image_array.shape)
    # logger.info(hist(image_array))
    return arr_to_str(image_array)

@celery_app.task(base=NetTask, bind=True, acks_late=True)
def infer_region_remotely(
    self,
    data_dir,
    model,
    file_type,
    region=0,
    input_size=(256,256), 
    in_channels=1,
    orientations=(0,),
    classes_of_interest=(0, 1, 2),
    **kwargs,
):
    from neuralnets.util.validation import segment
    from neuralnets.data.datasets import SlidingWindowDataset

    self.update_state(state="PROGRESS", meta=create_meta(1, 10))
    model = ROOT_DATA_FOLDER / model
    data_dir = ROOT_DATA_FOLDER / data_dir
        
    logger.info(f"model {model}")
    assert model.is_file()
    if self.model_path != model:
        self.model_path = model
        # remove possible previous net of different model
        # TODO keep multiple models in memory at the same time
        del self.net

    logger.info(f"data_dir {data_dir}")
    assert data_dir.is_dir()

    input_shape = (1, input_size[0], input_size[1])

    file_type = FileType(file_type)
    if not file_type.is_dir():
        try:
            # use first file in raw/ as data 
            data_dir = list(data_dir.iterdir())[0]
        except Exception as e:
            logger.error(f"Could not find data for data_dir={data_dir} and file_type={file_type}")
            return

    logger.info('Reading dataset')
    data = SlidingWindowDataset(str(data_dir), input_shape=input_shape, type=file_type.value, in_channels=in_channels)
    logger.info('Loading model')
    net = self.net
    logger.info('Segmenting')
    self.update_state(state="PROGRESS", meta=create_meta(2, 10))
    segmentation = segment(data.data[0][region], net, input_size, train=False, in_channels=in_channels, device=0, orientations=orientations)
    self.update_state(state="PROGRESS", meta=create_meta(9, 10))
    image_array = prob_to_label(segmentation, classes_of_interest)
    return arr_to_str(image_array)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)

    data_dir = ROOT_DATA_FOLDER / "EM" / "EMBL" / "raw"

    def get_data(x):
        arr = plt.imread(data_dir / f"data_{x:04d}.png")
        return arr

    print('test')

    region = get_data(0)
    input_size=[512, 512]
    # region = np.ones(shape=input_size)

    print(hist(region)[:10])
    classes_of_interest=[0, 1, 2]
    model = "models/mito_er_2d.pytorch"
    image_str = infer_region(region=arr_to_str(region), model_path=model, input_size=input_size, classes_of_interest=classes_of_interest)
    image_array = str_to_arr(image_str)
    print(image_array.shape)
    print(hist(image_array))
    