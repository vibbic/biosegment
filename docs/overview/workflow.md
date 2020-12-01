The following diagram gives on overview of the BioSegment workflow. Users interact with a frontend using their browser. They can visualize a dataset, edit annotations and create segmentations using AI models. The BioSegment backend handles the tasks given be the frontend and fetches the datasets from disk storage. For long-running tasks like conversion and fine-tuning, seperate workers are used.

![Placeholder](/assets/overview_v2.png)

## Data

- Dataset
    - Electron-microscopy data
    - example formats: pngseq, tif3d
- Topic of interest
    - example topics of interest: mitochondria, endoplasmatic reticulum...
- Segmentations
    - Attribution for each part of a dataset to a topic interest
    - mostly ground-truth or machine made
- Annotations
    - Stroke or area of a part of the dataset that is attributed
    - made by a human
- AI models
    -  able to take EM data and an annotation and create a segmentation
    -  can be pretrained and further fine-tuned with additional annonotations
    -  e.g. UNet


## Actors

- Scientist
    - A domain expert that wants to visualize and annotatate EM data with a specialized tool
- AI engineer
    - Implements and pretrains AI models

## User flow

Example of the user flow for a scientist when interacting with a BioSegment frontend.

![Placeholder](/assets/user_flow.png)