FROM russelljarvis/neuronunit
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN sudo rm -rf /opt/conda/lib/python3.5/site-packages/neuronunit-0.1.8.8-py3.5.egg/neuronunit
RUN sudo rm -rf $HOME/neuronunit
RUN sudo chown -R jovyan $HOME
COPY . $HOME/neuronunit
RUN sudo /opt/conda/bin/pip3 install -e $HOME/neuronunit
RUN sudo /opt/conda/bin/pip3 install coveralls

RUN sudo chown -R jovyan $HOME
WORKDIR $HOME/neuronunit/unit_test
RUN cat $HOME/neuronunit/unit_test/testNEURONparallel.py
RUN sudo chown -R jovyan $HOME
#RUN git clone https://github.com/ipython/ipyparallel
#WORKDIR ipyparallel
RUN sudo /opt/conda/bin/pip3 install ipyparallel
#COPY func2rc.sh .
#RUN func2rc.sh
#RUN sudo /opt/conda/bin/python3 setup.py install
COPY util.py /opt/conda/lib/python3.5/site-packages/ipyparallel/util.py
WORKDIR $HOME/neuronunit/unit_test
RUN sudo /opt/conda/bin/pip3 install pyNN
#ENTRYPOINT ipcluster start -n 8 --profile=default & sleep 25; ipython -m unittest testNEURONparallel.py
