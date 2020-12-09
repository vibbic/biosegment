# Workers

Long-running tasks need to be delegated to workers. [Celery](https://docs.celeryproject.org/en/stable/index.html) is used to manage these.

## backend-worker

Creates segmentations/annotations at the end of GPU worker task.

## GPU worker

GPU support in docker-compose is very experimental, not working currently
- see `docker-compose_gpu.yml`
- docker-compose override gives errors, that's why one .yml file is needed
- NVIDIA driver still isn't visible then, waiting for stable support

Current workaround
- expose rabbitMQ queue in docker-compose to host
- run celery worker on host without virtualization

```bash
cd gpu_worker

# install environment for neuralnets celery worker
conda env update -f celery_all_environment.yaml
conda activate celery_neuralnets

# On Linux
bash start_worker.sh

# On Windows
set ROOT_DATA_FOLDER=X:/biosegment/data
start_worker.bat   # On Windows
```

If force stopping the auto-reloading watchdog for workers (x2 Ctrl-C), some workers may linger.
This will show up as warning when a new worker with the same name is started.

View all host celery workers
```bash
ps aux|grep 'celery worker'
```

Kill them all
```bash
ps auxww | grep 'celery worker' | awk '{print $2}' | xargs kill -9
```
