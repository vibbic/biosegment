FROM continuumio/miniconda3

WORKDIR /code

COPY environment.yaml .

# Create the environment:
RUN /opt/conda/bin/conda env create -n biosegment -f environment.yaml 

ENV PATH /opt/conda/envs/biosegment/bin:$PATH