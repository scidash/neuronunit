# FROM russelljarvis/neuronunit
FROM scidash/scipy-notebook-plus
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN pip install dask
RUN pip install distributed
RUN pip install tornado
RUN sudo apt-get update
# RUN sudo apt-get install -y graphviz
# RUN conda install -y bokeh -c bokeh
RUN pip install pyzmq
# RUN pip install graphviz
RUN /opt/conda/bin/pip install git+https://github.com/scidash/sciunit@dev
RUN git clone https://github.com/vrhaynes/AllenInstituteNeuroML.git
RUN sudo /opt/conda/bin/pip install git+https://github.com/OpenSourceBrain/OpenCortex
COPY . /home/jovyan/neuronunit
WORKDIR /home/jovyan/neuronunit
RUN sudo /opt/conda/bin/pip install -e .
WORKDIR /home/jovyan/neuronunit/neuronunit/unit_test

#ENTRYPOINT python grid_entry_point.py
ENTRYPOINT /bin/bash

# WORKDIR $HOME
