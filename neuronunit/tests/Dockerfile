# author Russell Jarvis rjjarvis@asu.edu
# author Rick Gerkin rgerkin@asu.edu
# neuronunit-showcase

FROM scidash/neuronunit

#################
# Showcase for model optimization using NeuronUnit.
#################
USER root

RUN apt-get update
RUN apt-get install -y gvfs-bin libxss1 python3-setuptools
RUN apt-get install -y python-tk curl apt-utils


RUN chown -R jovyan $HOME
RUN chown -R jovyan /opt/conda/lib/python3.5/site-packages/

USER $NB_USER
RUN pip install --upgrade pip
RUN pip install -U matplotlib
RUN pip install IPython \
                jupyterhub \
                notebook \
                ipykernel \
                ipyparallel \
                enum34 \
		pypandoc \
		setuptools
RUN conda install mpi4py
##### Scoop #####

RUN easy_install gevent
RUN easy_install greenlet
WORKDIR $HOME/git
ADD . $HOME/git/scoop
WORKDIR scoop/scoop
RUN sudo python setup.py install



#RUN if [ ! -d "$HOME/git/scoop" ]; then \
#RUN git clone -b master https://github.com/russelljjarvis/scoop.git \
#cd scoop \
#    sudo python setup.py install \
#    else:


#fi


WORKDIR $HOME
RUN git clone https://github.com/russelljjarvis/scoop.git
WORKDIR scoop
RUN python setup.py install

##### DEAP ######
WORKDIR $HOME
RUN git clone https://github.com/DEAP/deap.git
WORKDIR deap
RUN python setup.py install

##### BluePyOpt ######
WORKDIR $HOME
RUN git clone https://github.com/russelljjarvis/BluePyOpt.git
WORKDIR BluePyOpt
RUN python setup.py install
RUN python -c "import bluepyopt"

##### IPython Cluster #####
USER $NB_USER
WORKDIR $HOME
RUN ipython profile create default
RUN pip install git+https://github.com/roryk/ipython-cluster-helper
# Must be done as root to create directories in /usr/local
RUN sudo /opt/conda/bin/ipcluster nbextension enable

RUN printenv PATH
RUN python -c "import pyneuroml"
RUN python -c "import neuronunit"
RUN python -c "from neuronunit.models.reduced import ReducedModel"
RUN conda install quantities
RUN python -c "import quantities"
RUN python -c "import neuron"
RUN python -c "import pyneuroml"
RUN nrnivmodl
RUN python -c "import scoop"
RUN python -c "import deap"
RUN nrniv

RUN sudo /opt/conda/bin/pip install sphinx
RUN sudo git clone https://github.com/russelljjarvis/scoop.git
RUN sudo /opt/conda/bin/python setup.py install
#RUN export PATH=$PATH

#uncomment the following line to test ipcluster during build.
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
#RUN /opt/conda/bin/ipython3 -c "import sys; print(sys.path)"
#RUN /opt/conda/bin/ipython3 -c "import os; print(sys.path)"



RUN echo "backend      : Agg" >> /opt/conda/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc

WORKDIR /home/mnt

RUN /opt/conda/bin/ipython3 -c "from ipyparallel import Client; import os; os.system('ipcluster start --profile=chase --debug &')"
# /opt/conda/bin/python3 exhaustive_search.py
RUN sudo pip install -y NotebookTools
RUN conda install anaconda-client
WORKDIR $HOME/git
