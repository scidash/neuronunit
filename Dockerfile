FROM russelljarvis/neuronunit
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN sudo rm -rf /opt/conda/lib/python3.5/site-packages/neuronunit-0.1.8.8-py3.5.egg/neuronunit
RUN sudo rm -rf $HOME/neuronunit
#RUN sudo chown -R jovyan $HOME
COPY . $HOME/neuronunit


RUN sudo chown -R jovyan $HOME
RUN pip uninstall -y pyneuroml
#RUN pip uninstall -y pylems
RUN pip install -e $HOME/neuronunit --ignore-installed --process-dependency-links
RUN pip install lazyarray pyNN
RUN pip install dask
RUN pip install distributed

COPY util.py /opt/conda/lib/python3.5/site-packages/ipyparallel/util.py
RUN pip install tensorflow
WORKDIR $HOME
RUN git clone https://github.com/calvinschmdt/EasyTensorflow.git easy_tensorflow
WORKDIR easy_tensorflow

RUN sudo apt-get install -y python-setuptools
RUN sudo python setup.py install
WORKDIR $HOME

COPY $HOME/git/BluePyOpt ~/HOME/BluePyOpt
RUN pip install -e $HOME/BluePyOpt
WORKDIR $HOME
RUN pip install prospector
RUN pip install pyosf

# RUN sed -i.bak '41d' /opt/conda/lib/python3.5/site-packages/lems/model/model.py


#RUN sudo /opt/conda/bin/pip3 install coveralls
#RUN sudo rm /opt/conda/lib/python3.5/site-packages/PyLEMS-0.4.9-py3.5.egg

#WORKDIR $HOME/neuronunit/unit_test
#RUN sudo chown -R jovyan $HOME
#COPY util.py /opt/conda/lib/python3.5/site-packages/ipyparallel/util.py
#WORKDIR $HOME/neuronunit/unit_test/NeuroML2
#RUN nrnivmodl
#WORKDIR $HOME/neuronunit/unit_test
#ENTRYPOINT ipcluster start -n 8 --profile=default & sleep 15; python stdout_worker.py & /bin/bash
# ipython -i test_exhaustive_search.py
