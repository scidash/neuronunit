FROM russelljarvis/neuronunit
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
COPY . $HOME/neuronunit

RUN pip install dask
RUN pip install distributed
RUN pip install tornado
RUN sudo apt-get update
#RUN sudo apt-get install -y graphviz
#$RUN conda install -y bokeh -c bokeh
RUN pip install pyzmq
RUN pip install graphviz
RUN /bin/bash -c exec dask-worker scheduler:8786
RUN /opt/conda/bin/pip install git+https://github.com/scidash/sciunit@dev
RUN /opt/conda/bin/pip install pinstall graphviz
RUN sudo /opt/conda/bin/pip uninstall -y sciunit
RUN sudo /opt/conda/bin/pip install git+https://github.com/scidash/sciunit@dev
RUN git clone https://github.com/vrhaynes/AllenInstituteNeuroML.git
RUN sudo /opt/conda/bin/pip install git+https://github.com/OpenSourceBrain/OpenCortex@dev


WORKDIR $HOME
