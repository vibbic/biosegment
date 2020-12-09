# Viewer

The viewer is an interface implemented using [Dash](https://plotly.com/dash/). It can be used for basic visualizations of BioSegment data. The viewer also allows for annotating, running a segmentation job and fine-tuning a model.

- Open a dataset via the Dashboard or go to the URL directly e.g. `http://localhost/dash/viewer`.
- Use the refresh buttons to update the dropdowns with backend data.

The viewer has 4 components.

## Viewer

- Given a dataset, segmentations and slider location, a view of a slice of the dataset will be shown. Differently annotated regions will be colored.
- the slider is the `z` coordinate.
- Multiple segmentations can be selected at the same time.

## Annotations tools

- classes of interest
    - e.g. mitochondria, ER
    - determined by dataset
- Stroke width of the annotation brush
- Selected annotation
- Create a new annotation for given name
    - when no annotation is selected, will create an empty annotation
    - when an annotation is selected, will copy the annotation and give it the new name.
- Start editing current annotation
    - The viewer will update with annotation tools
    - when done, click on the done button to save the annotation. Only then will the annotation be saved in the backend.

## Run segmentation

- create a segmentation for currently selected dataset, given model and new segmentation name

## Fine-tune model

- create a new model from an existing model, given annotation and new model name. Epochs determines the amount of fine-tuning.
