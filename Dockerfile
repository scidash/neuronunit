FROM russelljarvis/neuronunit
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN sudo rm -rf /opt/conda/lib/python3.5/site-packages/neuronunit-0.1.8.8-py3.5.egg/neuronunit
RUN sudo rm -rf $HOME/neuronunit
COPY . $HOME/neuronunit

RUN pip install dask
RUN pip install distributed
#COPY BluePyOpt ~/HOME/BluePyOpt
#RUN pip install -e $HOME/BluePyOpt
WORKDIR $HOME



