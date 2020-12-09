# Data folder

BioSegment works with datasets located in a `ROOT_DATA_FOLDER` folder, mostly a mounted network drive. Users can access the network drive directly to import or export files with their own tools and workflows.

## Structure

- ROOT_DATA_FOLDER
    - setup.json
        - JSON file containing configurations the backend reads during initialization of the database. Not required. Used in development. See the example at `scripts/example_setup.json`.
    - EM/
        - {dataset_name} e.g. EMBL
            - raw/
                - {pngs}
    - models/
        - {model_name} e.g. unet_2d
            - = output folder of neuralnets training
            - saved model e.g. best_checkpoint.pytorch
    - segmentations/
        - {dataset_name}
            - labels/
                - = ground truth labels of the dataset
                - {pngs}
            - {segmentation_name}
                - = output folder of neuralnets inference
                - {pngs}
    - annotations/
        - {dataset_name}
            - {annotation_name}
                - saved annotations e.g. annotations.json

## New dataset

Adding a new dataset requires creating a folder with the dataset title in the `EM/` folder e.g. `{ROOT_DATA_FOLDER}/EM/{NEW_DATASET_NAME}`. Then the dataset files need to be set in this folder e.g. a single `.tiff` or multiple `.png` files.
