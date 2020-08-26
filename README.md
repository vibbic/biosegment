# Biosegment backend

## Install
```
conda env create -f environment.yaml --prune
conda activate biosegment
```

## Run development
```
uvicorn main:backend_app --reload
```