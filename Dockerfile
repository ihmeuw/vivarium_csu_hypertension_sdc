FROM continuumio/miniconda3

# # setup an arbitrary non-root user
# RUN groupadd -g 5150 vivarium && \
#     useradd -r -u 5150 -g vivarium vivarium && \
#     chown -R 5150:5150 /opt/conda

# WORKDIR /home/vivarium
# USER vivarium

WORKDIR /vivarium

# copy in minimal required files
COPY setup.py ./
COPY README.rst ./
COPY src/ ./src/

RUN conda install hdf5
# && pip install .
