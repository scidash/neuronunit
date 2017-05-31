#author Russell Jarvis rjjarvis@asu.edu
#author Rick Gerkin rgerkin@asu.edu
FROM scidash/neuronunit

USER root


RUN pip install --upgrade pip
RUN conda install matplotlib

RUN easy_install gevent
RUN easy_install greenlet
RUN apt-get update
RUN apt-get install -y gvfs-bin
#I cant remember what gvfs-bin does
RUN apt-get install -y libxss1
RUN apt-get install -y curl apt-utils
RUN apt-get install -y python3-setuptools
RUN chown -R $NB_USER $HOME
RUN chown -R $NB_USER /opt/conda/lib/python3.5/site-packages/


RUN easy_install gevent
RUN easy_install greenlet
WORKDIR $HOME
RUN git clone https://github.com/soravux/scoop
WORKDIR scoop
RUN python setup.py install
WORKDIR $HOME
RUN git clone https://github.com/DEAP/deap.git
WORKDIR deap
RUN python setup.py install

RUN apt-get install -y python3-setuptools
ENV PATH $PATH:/opt/conda/bin
RUN conda install mpi4py
RUN pip install \
    deap \
    IPython \
    jupyterhub \
    notebook \
    ipykernel \
    ipyparallel \
    enum34
RUN ipython profile create default

#WORKDIR $HOME
#RUN git clone https://github.com/russelljjarvis/BluePyOpt.git
#WORKDIR BluePyOpt
#RUN python setup.py install
#This line python-tk is a problem
#RUN apt-get install -y python-tk
#RUN chown -R jovyan $HOME
#HEALTHCHECK python -c "import bluepyopt"

RUN conda install mpi4py
RUN pip install git+https://github.com/roryk/ipython-cluster-helper

RUN sudo /opt/conda/bin/ipcluster nbextension enable

RUN apt-get install -y curl apt-utils
RUN apt-get update

WORKDIR $HOME

RUN chown -R jovyan /opt/conda/lib/python3.5/site-packages/

RUN pip install git+https://github.com/roryk/ipython-cluster-helper

RUN ipcluster nbextension enable
#RUN chmod 755 jovyan/ipython/profile_default
#RUN chmod 744 jovyan/ipython/profile_default
USER $NB_USER


ENV PATH $PATH:/opt/conda/bin
ENV PATH $PATH:/opt/conda/bin/ipcluster
ENV PATH $PATH:/opt/conda/bin/ipython
ENV PATH $PATH:/opt/conda/bin/pip
ENV PATH $PATH:/opt/conda/bin/python
ENV PATH $PATH:/opt/conda/lib/python3.5/site-packages
ENV PATH $PATH:$PYTHONPATH

RUN echo $PATH
HEALTHCHECK CMD python -c "import pyneuroml"
HEALTHCHECK CMD python -c "import neuronunit"
HEALTHCHECK CMD python -c "from neuronunit.models.reduced import ReducedModel"
HEALTHCHECK CMD python -c "from deap.tools import hypervolume"
HEALTHCHECK CMD python -c "import quantities"
HEALTHCHECK CMD python -c "import neuron"
HEALTHCHECK CMD python -c "import pyneuroml"
HEALTHCHECK CMD nrnivmodl
HEALTHCHECK CMD python -c "import scoop"
HEALTHCHECK CMD python -c "import deap"
HEALTHCHECK CMD nrniv

RUN export PATH=$PATH

# Use this NeuronUnit (delete later)
USER root
ARG MOD_DATE=0
RUN echo $MOD_DATE
# Make neuronunit source directory in Travis image visible to Docker.
#ADD . $HOME/neuronunit/neuronunit/tests
#RUN chown -R $NB_USER $HOME/neuronunit/neuronunit/tests
USER $NB_USER

#fun comment the following line to test ipcluster during build.
#RUN ipcluster start --profile=chase --debug &
#https://github.com/russelljjarvis/neuronunit.git
WORKDIR $HOME
ENV PYTHONPATH $HOME/neuronunit:$PYTHONPATH
RUN  if [ ! -d "$/home/jovyan/git/neuronunit" ]; then git clone -b dev https://github.com/russelljjarvis/neuronunit.git; fi
WORKDIR $HOME/neuronunit/neuronunit/tests/NeuroML2

RUN nrnivmodl
RUN ls LEMS_2007One.xml
WORKDIR $HOME/neuronunit/neuronunit/tests/
RUN ls *.pickle

WORKDIR /home/git
USER root
RUN chown -R $NB_USER .
USER $NB_USER

# This ARG can be replaced with a build-arg containing the hash of the current
# HEAD, which will ensure that if the HEAD has changed (i.e. new commits),
# then a new tar of the repo will be downloaded.
ARG HEAD=1

WORKDIR $HOME
WORKDIR sciunit
RUN sudo git clone -b dev https://github.com/russelljjarvis/sciunit.git
#RUN tar -xvzf sciunit.tar.gz -C sciunit --strip-components=1
WORKDIR sciunit
RUN sudo python setup.py install
RUN pip install cypy
RUN python test_all.py
RUN conda install plotly seaborn
RUN pip install prospector
WORKDIR /home/git/neuronunit
#RUN python -c "from prospector import run; run.main()"
#run.main()

RUN sudo chown -R jovyan .

RUN echo "import os">> run_ipp1.p

#RUN echo "os.system('ipcluster start --profile=chase --debug &')" >> run_ipp1.py
#RUN echo "import ipyparallel as ipp ; rc = ipp.Client(profile='chase'); print('hello from before cpu '); print(rc.ids);" >> run_ipp2.py
#RUN ipython run_ipp1.py && sleep 15 && ipython run_ipp2.py
HEALTHCHECK CMD /opt/conda/bin/ipython3 -c "import sys; print(sys.path)"
#RUN /opt/conda/bin/ipython3 -c "import os; print(sys.path)"

HEALTHCHECK CMD /opt/conda/bin/ipython3 -c "from ipyparallel import Client; import os; os.system('ipcluster start --profile=chase --debug &')"
# /opt/conda/bin/python3 exhaustive_search.py
RUN echo "alias es='/opt/conda/bin/python3 exhaustive_search.py'" >> ~/.bashrc
ENV QT_QPA_PLATFORM offscreen #&& /opt/conda/bin/ipython exhaustive_search.py 