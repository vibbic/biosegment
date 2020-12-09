# Tutorial

## Run a segmentation on a new dataset

### Add a new dataset

1. Create a folder with the dataset title at `ROOT_DATA_FOLDER/EM/`. e.g. `ROOT_DATA_FOLDER/EM/DATASET_TITLE`.
2. Create a folder titled `raw` in this new folder.
3. Move the files (pngs/tiff) to the `raw` folder.
      1. pngs should be labeled `{prefix}{index}.png`, with `index` ranging from `0000` to e.g. `0049` for a dataset with resultion z = `50`. Note that indexing starts at 0. The prefix can be empty and should be the same for all files to ensure the sorted order.
      2. a tiff file should use the suffix `.tif`. A certain filename is not required.
4. Navigate to the dataset create frontend e.g. `http://localhost/main/datasets/create`.
5. Fill in the form and save
      1. choose file type `pngseq` for multiple png files and `tif3d` for a single tiff file.
      2. The resulution of the files can be obtained with e.g. [Fiji](https://fiji.sc/)`Image > Show Info...`. Width == x, Height == y and Depth == z. For `pngseq`, `z` is also equal to the number of png files.
6. The dataset should show up in the list and be accessible by the viewer.

### Open the dataset in the viewer

1. Navigate to `http://localhost/dash/viewer`.
2. Use `Selected dataset` to show the new dataset.

### Run a segmentation using the correct model

1. Use `Selected model` to pick the correct pretrained model. e.g. `mito 2D` when only  mitochondria are a class of interest.
2. Choose a descriptive segmentation.
3. Click `Start new segmentation`. The loading bar should show success after <1min.

### View the generated segmentation

1. Click the Refresh button at `Selected segmentation` in the Viewer.
2. Selected the new segmentation with the dropdown.

### Access the files of the segmentation

1. The segmentation should be visible using the frontend e.g. `http://localhost/main/segmentations/all`. In the edit dialogue, the location of the segmentation is visible. You can navigate to this folder on the network disk e.g. `ROOT_DATA_FOLDER/segmentations/{DATASET_TITLE}/{SEGMENTATION_TITLE}/`.

## Add an external annotation to a dataset

1. Create a folder at `annotations/{DATASET_TITLE}/` with the annotation title e.g. `ANNOTATION_TITLE`.
2. Add the annotation files to the folder
3. Create a new annotation via the frontend e.g. `http://localhost/main/annotations/create`.
      1. location = e.g. `annotations/{DATASET_TITLE}/ANNOTATION_FOLDER`
      2. TODO type = ...

To work from a previous annotation, duplicate the folder with a different name or reexport to a different folder from your external annotation tool. Then add it via the frontend as usual.

## Fine-tune a model with an annotation

### Create a new annotation

Use the annotation tools or add an external annotation.

### Fine-tune a model using the created annotation

1. In Fine-tune model
      1. select the model and annotation
      2. epochs = amount of iterations
      3. choose a new model name e.g. `FINETUNED_MODEL_MITO`
2. Click Retrain model. The loading bar should show success after 10min-1h.

### Run a segmentation using the fine-tuned model

1. Click refresh at Run segmentation > selected model.
2. Select the model `FINETUNED_MODEL_MITO`, choose a new segmentation name and run.

Now you can view the segmentation as shown in a different tutorial.
