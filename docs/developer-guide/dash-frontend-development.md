# Dash frontend development

See [Dash Python docs](https://plotly.com/dash/) for more information.

The dataset viewer is implemented using Plotly.
## Structure

- `dash_frontend/app/`
    - `api/`
        - Python implementation of the backend API
    - `assets/`
        - custom CSS code
    - `components/`
        - a Dash component has a unique id, a layout and decorated Python functions
    - `pages/`
        - contains pages that import components
    - `index.py`
        - imports pages and routes them
    - `DatasetStore.py`
        - contains Datasets
    - `Dataset.py`
        - uses API to list backend data. Read file system based on location string in backend.
        - Uses [skimage.ImageCollection](https://scikit-image.org/docs/0.7.0/api/skimage.io.collection.html)