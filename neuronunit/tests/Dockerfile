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
#RUN apt-get install -y python-tk


RUN chown -R jovyan $HOME
#RUN python -c "import bluepyopt"

RUN conda install mpi4py
#RUN pip install git+https://github.com/roryk/ipython-cluster-helper

#RUN ipcluster nbextension enable

RUN apt-get install -y curl apt-utils
RUN apt-get update

WORKDIR $HOME

RUN chown -R jovyan /opt/conda/lib/python3.5/site-packages/

USER $NB_USER


ENV PATH $PATH:/opt/conda/bin
ENV PATH $PATH:/opt/conda/bin/ipcluster
ENV PATH $PATH:/opt/conda/bin/ipython
ENV PATH $PATH:/opt/conda/bin/pip
ENV PATH $PATH:/opt/conda/bin/python
ENV PATH $PATH:/opt/conda/lib/python3.5/site-packages
ENV PATH $PATH:$PYTHONPATH

RUN echo $PATH
RUN python -c "import pyneuroml"
RUN python -c "import neuronunit"
RUN python -c "from neuronunit.models.reduced import ReducedModel"
RUN python -c "from deap.tools import hypervolume"
RUN python -c "import quantities"
RUN python -c "import neuron"
RUN python -c "import pyneuroml"
RUN nrnivmodl
RUN python -c "import scoop"
RUN python -c "import deap"
RUN nrniv
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
RUN git clone -b dev https://github.com/russelljjarvis/neuronunit.git
WORKDIR $HOME/neuronunit/neuronunit/tests/NeuroML2

RUN nrnivmodl
RUN ls LEMS_2007One.xml
WORKDIR $HOME/neuronunit/neuronunit/tests/
RUN ls *.pickle


WORKDIR /home/mnt
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

RUN pip install vulture
#RUN zsh
WORKDIR /home/mnt
#ENTRYPOINT zsh
#ENTRYPOINT python nsga.py
#RUN sudo apt-get install -y zsh
#WORKDIR $WORK
#RUN sudo git clone https://github.com/jendrikseipp/vulture
#WORKDIR vulture
#RUN ls *
#RUN cat README.rst
#RUN sudo python install.py
