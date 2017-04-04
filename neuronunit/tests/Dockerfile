#author Russell Jarvis rjjarvis@asu.edu
#author Rick Gerkin rgerkin@asu.edu
FROM scidash/neuronunit
USER root


RUN pip install --upgrade pip
RUN pip install matplotlib

RUN easy_install gevent
RUN easy_install greenlet
RUN apt-get update
RUN apt-get install -y gvfs-bin
#I cant remember what gvfs-bin does
RUN apt-get update



ENV WORK_HOME $HOME/work/scidash
RUN ipython profile create default
RUN apt-get install -y libxss1
RUN git clone https://github.com/soravux/scoop
WORKDIR scoop
RUN python setup.py install
RUN apt-get install -y python3-setuptools
ENV PATH $PATH:/opt/conda/bin
RUN easy_install gevent
RUN easy_install greenlet

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

RUN sudo /opt/conda/bin/pip install \
	IPython \
	jupyterhub \
	notebook \
	ipykernel \
	ipyparallel \
	enum34


RUN sudo printenv PATH
RUN python -c "import pyneuroml"
RUN python -c "import neuronunit"
RUN python -c "from neuronunit.models.reduced import ReducedModel"
RUN python -c "import quantities"
RUN python -c "import neuron"
RUN python -c "import pyneuroml"
RUN nrnivmodl
RUN python -c "import scoop"
RUN python -c "import deap"

RUN nrniv

RUN export PATH=$PATH

#un comment the following line to test ipcluster during build.
#RUN ipcluster start --profile=chase --debug &
#https://github.com/russelljjarvis/neuronunit.git
WORKDIR $HOME

RUN echo "do this part again such that pickled files can be used to bypass problems with aibs.py in the short term !"

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
#RUN echo "down load anyway please"
#RUN mkdir -p sciunit
WORKDIR sciunit
RUN git clone -b dev http://github.com/scidash/sciunit
#RUN tar -xvzf sciunit.tar.gz -C sciunit --strip-components=1
WORKDIR sciunit
RUN python setup.py install
RUN python test_all.py
RUN git pull origin dev
RUN python setup.py install
WORKDIR /home/mnt
ENTRYPOINT python nsga.py
