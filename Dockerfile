#author Russell Jarvis rjjarvis@asu.edu
#author Rick Gerkin rgerkin@asu.edu

FROM scidash/neuron-mpi-neuroml

USER root




#Install rgerkin version of AllenSDK

RUN pip install neo
RUN pip install elephant
RUN pip install bs4
RUN pip install quantities
RUN pip install execnet
RUN pip install git+https://github.com/soravux/scoop
RUN pip install git+https://github.com/DEAP/deap
RUN pip install git+https://github.com/rgerkin/rickpy


RUN pip install --upgrade pip
RUN apt-get install -y libxml2-dev libxslt1-dev

ENV WORK_HOME $HOME/work/scidash
WORKDIR $HOME/work/scidash
RUN git clone https://github.com/NeuroML/pyNeuroML
WORKDIR pyNeuroML
RUN python setup.py install
ENV PYTHONPATH=$PYTHONPATH:$WORK_HOME/pyNeuroML/neuroml
RUN python -c "import neuroml"


RUN git clone -b dev https://github.com/scidash/sciunit.git $WORK_HOME/sciunit
ENV PYTHONPATH=$PYTHONPATH:$WORK_HOME/sciunit
RUN python -c "import sciunit"



RUN pip install git+https://github.com/rgerkin/AllenSDK@python3.5 --process-dependency-links
RUN python -c "import pyneuroml"

RUN conda install -y pyqt
ENV WORK_HOME $HOME/work/scidash


#RUN git clone https://github.com/rgerkin/IzhikevichModel $WORK_HOME/izk

RUN git clone -b dev https://github.com/scidash/neuronunit.git $WORK_HOME/neuronunit
ENV PYTHONPATH=$PYTHONPATH:$WORK_HOME/neuronunit
RUN python -c "import neuronunit"
RUN python -c "from neuronunit.models.reduced import ReducedModel"
RUN python -c "import quantities"
RUN python -c "import neuron"
RUN python -c "import pyneuroml"
RUN nrnivmodl 
RUN python -c "import scoop"
RUN python -c "import deap"
RUN nrniv

RUN apt-get update \
      && apt-get install -y sudo \
      && rm -rf /var/lib/apt/lists/*
RUN echo "jovyan ALL=NOPASSWD: ALL" >> /etc/sudoers

WORKDIR $HOME/work/scidash
RUN echo "REDO"
RUN git clone https://github.com/russelljjarvis/sciunitopt 
WORKDIR $HOME/work/scidash/sciunitopt
RUN ls
ENV PYTHONPATH=$PYTHONPATH:$HOME/work/scidash/sciunitopt 
WORKDIR $HOME
RUN python -c "import sciunitopt"
RUN python -c "from sciunitopt.deap_config_simple_sum import DeapCapsule"

RUN chown -R jovyan $HOME

USER $NB_USER

#The following are convenience aliases
RUN echo 'alias nb="jupyter-notebook --ip=* --no-browser"' >> ~/.bashrc
RUN echo 'alias mnt="cd /home/mnt"' >> ~/.bashrc
RUN echo 'alias erc="emacs ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias src="source ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias egg="cd /opt/conda/lib/python3.5/site-packages/"' >> ~/.bashrc 
RUN echo 'alias nu="cd /home/jovyan/work/scidash/neuronunit"' >> ~/.bashrc
RUN echo 'alias model="cd /work/scidash/neuronunit/neuronunit/models"' >> ~/.bashrc
RUN echo 'alias sciunit="cd /work/scidash/sciunit"' >> ~/.bashrc
RUN echo 'alias nu="python -c "from neuronunit.models.reduced import ReducedModel""'
RUN echo 'alias opt="python -i neuronunit/tests/Optimize.py"' >> ~/.bashrc


WORKDIR /home/mnt
#WORKDIR /home/jovyan/mnt/sciunitopt
#ENTRYPOINT python -i /home/jovyan/mnt/sciunitopt/AIBS.py 


RUN git config --global user.name "Russell Jarvis"
RUN git config --global user.email "rjjarvis@asu.edu; bash"
