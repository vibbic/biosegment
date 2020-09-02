FROM continuumio/miniconda3:4.8.2

WORKDIR /backend

COPY celery_pytorch_environment.yaml .

RUN /opt/conda/bin/conda env update -n my-env \
    -f celery_pytorch_environment.yaml 
    # && /opt/conda/bin/conda clean -a

COPY celery_celery_environment.yaml .

RUN /opt/conda/bin/conda env update -n my-env \
    -f celery_celery_environment.yaml 
    # && /opt/conda/bin/conda clean -a

COPY celery_neuralnets_environment.yaml .

RUN /opt/conda/bin/conda env update -n my-env \
    -f celery_neuralnets_environment.yaml 
    # && /opt/conda/bin/conda clean -a

ENV PATH /opt/conda/envs/my-env/bin:$PATH

COPY app/app /app

# Clone the neuralnets repo
RUN git clone https://github.com/saeyslab/neuralnets.git

# Add python modele to backend app module
RUN mv neuralnets/neuralnets /app/

# Add neuralnets folder
# RUN git clone 

WORKDIR /app

ENV PYTHONPATH=/app

COPY ./app/worker-start.sh /worker-start.sh

RUN chmod +x /worker-start.sh

CMD ["bash", "/worker-start.sh"]
