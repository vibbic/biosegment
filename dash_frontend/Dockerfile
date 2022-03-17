FROM continuumio/miniconda3

WORKDIR /app

COPY . .

# Create the environment:
RUN /opt/conda/bin/conda env create -n biosegment -f environment.yaml 

ENV PATH /opt/conda/envs/biosegment/bin:$PATH

ENV PYTHONPATH=/app