FROM jupyter/scipy-notebook

USER root

# install necessary dependencies for Cityflow and SUMO under user directory
RUN apt-get update && apt-get install -y build-essential cmake wget git && \
    apt-get install -y g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig

ENV SUMO_HOME="$PWD/sumo"
RUN conda create -n libsignal python=3.8 -y && source activate && conda activate libsignal && git clone https://github.com/cityflow-project/CityFlow.git && cd CityFlow && pip install . &&\
    cd ../ && git clone --recursive https://github.com/eclipse/sumo && mkdir sumo/build/cmake-build && cd sumo/build/cmake-build && \
    cmake ../.. && make -j$(nproc)

ENV  SUMO_HOME=$HOME/sumo
ENV PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

RUN cd /home/jovyan/work && source activate && conda activate libsignal && git clone --recursive https://github.com/DaRL-LibSignal/LibSignal.git && chmod -R 777 . &&\
    pip install ipykernel && python -m ipykernel install --name libsignal && pip install torch && pip install setuptools==65.5.0 && pip install gym lmdb mpmath==1.2.1 PyYAML==6.0 numpy &&\
    pip install pfrl torch_geometric torch_scatter matplotlib

USER ${NB_UID}

WORKDIR ${HOME}