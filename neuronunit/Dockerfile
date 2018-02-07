FROM russelljarvis/neuronunit
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN sudo rm -rf /opt/conda/lib/python3.5/site-packages/neuronunit-0.1.8.8-py3.5.egg/neuronunit
RUN sudo rm -rf $HOME/neuronunit
COPY . $HOME/neuronunit
RUN sudo /opt/conda/bin/pip3 install -e $HOME/neuronunit
COPY util.py /opt/conda/lib/python3.5/site-packages/ipyparallel/util.py
RUN sudo /opt/conda/bin/pip3 install coveralls
COPY func2rc.sh .
RUN sudo chown -R jovyan $HOME
COPY neuronunit $HOME/neuronunit
RUN cat $HOME/neuronunit/setup.py
#RUN bash func2rc.sh
WORKDIR $HOME/neuronunit
RUN sudo chown -R jovyan $HOME
RUN pip install prospector
RUN pip install pyosf

#ENTRYPOINT ipcluster start -n 8 --profile=default & sleep 10; ipython -m unittest testNEURONparallel.py
