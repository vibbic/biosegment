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

## Testing
Have pytest installed
```
rm test.db; pytest
```

Warning "...aiofiles/os.py:10: DeprecationWarning: "@coroutine"..." is ok, should be solved upstream.