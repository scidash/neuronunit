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
RUN pip uninstall -y pylems
RUN pip install -e $HOME/neuronunit --ignore-installed --process-dependency-links
RUN pip install lazyarray pyNN

COPY util.py /opt/conda/lib/python3.5/site-packages/ipyparallel/util.py
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
