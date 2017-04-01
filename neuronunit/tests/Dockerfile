#author Russell Jarvis rjjarvis@asu.edu
#author Rick Gerkin rgerkin@asu.edu
FROM scidash/neuronunit

USER root
RUN apt-get update
RUN apt-get install -y gvfs-bin 
#I cant remember what gvfs-bin does
RUN apt-get install -y libxss1
RUN apt-get install -y curl apt-utils
RUN apt-get install -y python3-setuptools

USER $NB_USER
WORKDIR $HOME
RUN easy_install gevent
RUN easy_install greenlet
RUN git clone https://github.com/soravux/scoop
WORKDIR scoop
RUN python setup.py install
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
#RUN python -c "import bluepyopt"
#RUN pip install git+https://github.com/roryk/ipython-cluster-helper
#RUN ipcluster nbextension enable
#un comment the following line to test ipcluster during build.
#RUN ipcluster start --profile=chase --debug &
#https://github.com/russelljjarvis/neuronunit.git

RUN chown -R jovyan $HOME
RUN chown -R jovyan /opt/conda/lib/python3.5/site-packages/

WORKDIR $HOME

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
RUN python -c "import quantities"
RUN python -c "import neuron"
RUN python -c "import pyneuroml"
RUN nrnivmodl
RUN python -c "import scoop"
RUN python -c "import deap"
RUN nrniv
RUN export PATH=$PATH
RUN echo "do this part again such that pickled files can be used to bypass problems with aibs.py in the short term !"

# Make neuronunit source directory in Travis image visible to Docker.
ADD . $HOME/neuronunit
WORKDIR $HOME/neuronunit/neuronunit/tests/NeuroML2
RUN nrnivmodl
RUN ls LEMS_2007One.xml
WORKDIR $HOME/neuronunit/neuronunit/tests/
RUN ls *.pickle
ENTRYPOINT python -m scoop nsga.py
