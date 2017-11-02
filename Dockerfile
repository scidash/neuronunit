FROM russelljarvis/neuronunit
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN sudo rm -rf /opt/conda/lib/python3.5/site-packages/neuronunit-0.1.8.8-py3.5.egg/neuronunit
RUN sudo rm -rf $HOME/neuronunit
RUN sudo chown -R jovyan $HOME
RUN echo "redo this part 1"

#RUN sudo /opt/conda/bin/pip3 uninstall -y quantities
RUN sudo git clone https://github.com/python-quantities/python-quantities
WORKDIR python-quantities
RUN sudo /opt/conda/bin/python3 setup.py install
#RUN sudo /opt/conda/bin/pip3 install quantities

RUN sudo git clone -b dev https://github.com/scidash/sciunit
RUN sudo /opt/conda/bin/pip3 install -e sciunit
COPY . $HOME/neuronunit
RUN sudo /opt/conda/bin/pip3 install -e $HOME/neuronunit --process-dependency-links

COPY util.py /opt/conda/lib/python3.5/site-packages/ipyparallel/util.py
RUN sudo /opt/conda/bin/pip3 install coveralls
#COPY func2rc.sh .
RUN sudo chown -R jovyan $HOME
WORKDIR $HOME/neuronunit/unit_test
RUN cat $HOME/neuronunit/unit_test/testNEURONparallel.py
RUN sudo chown -R jovyan $HOME
#ENTRYPOINT ipcluster start -n 8 --profile=default & sleep 25; ipython -m unittest testNEURONparallel.py
