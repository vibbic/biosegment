# Dash frontend

Frontend for BioSegment using Dash

## Install
```
conda env create -f environment.yaml --prune
conda activate biosegment
```

## Run development
```
PYTHONPATH=app gunicorn app.index:server --reload
```
