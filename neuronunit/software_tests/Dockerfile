FROM scidash/neuron-mpi-neuroml

USER root
RUN pip install git+https://github.com/soravux/scoop
RUN pip install git+https://github.com/DEAP/deap
RUN pip install git+https://github.com/rgerkin/rickpy

WORKDIR /home/jovyan/work/scidash/pyNeuroML
RUN pip install git+https://github.com/NeuroML/pyNeuroML --process-dependency-links
RUN pip install neo elephant bs4

WORKDIR /home/jovyan/work/scidash
RUN pip install git+https://github.com/AllenInstitute/AllenSDK@py34_rgerkin --process-dependency-links

RUN pip install git+https://github.com/python-quantities/python-quantities
WORKDIR /home/jovyan/work/scidash/pyNeuroML
RUN pip install git+https://github.com/NeuroML/pyNeuroML --process-dependency-links
RUN python -c "import pyneuroml"

RUN conda install -y pyqt
RUN conda install -y matplotlib 


#The purpose of adding python packages via symbolic links as done below, is to make it such that the developer has write access to the py package
#and development changes are effective immediately to the py package.

WORKDIR /home/jovyan/work/scidash

RUN git clone -b dev https://github.com/russelljjarvis/sciunit.git
                     
WORKDIR /home/jovyan/work/scidash/sciunit
RUN ln -s /home/jovyan/work/scidash/sciunit/sciunit /opt/conda/lib/python3.5/site-packages/sciunit
RUN python -c "import sciunit"

WORKDIR /home/jovyan/work/scidash
RUN git clone -b dev https://github.com/russelljjarvis/neuronunit.git
RUN ln -s /home/jovyan/work/scidash/neuronunit/neuronunit /opt/conda/lib/python3.5/site-packages


RUN python -c "import neuronunit"
RUN python -c "from neuronunit.models.reduced import ReducedModel"
RUN python -c "import quantities, neuronunit, sciunit"



#Check if anything broke 
RUN python -c "import neuron; import sciunit; import neuronunit; import pyneuroml"
RUN nrnivmodl 
RUN python -c "import scoop; import deap"
RUN nrniv


#Install channelworm

RUN pip install django
#Note the code below is not sufficient to properly run channelworm in python
#channelworm depends on DJANGO which needs a web server configuration to run.
RUN git clone https://github.com/russelljjarvis/ChannelWorm.git 
RUN ln -s /home/jovyan/work/scidash/ChannelWorm/channelworm /opt/conda/lib/python3.5/site-packages/channelworm
RUN python -c "import channelworm"


WORKDIR /home/jovyan/work/scidash

#Install NeuroConstruct

RUN git clone https://github.com/NeuralEnsemble/neuroConstruct.git 
WORKDIR /home/jovyan/work/scidash/neuroConstruct
RUN bash nC.sh -make
RUN bash nCenv.sh
RUN echo 'export NC_HOME=home/jovyan/work/scidash/neuroConstruct' >> ~/.bashrc


RUN ln -s /home/jovyan/work/scidash/neuroConstruct/pythonnC /opt/conda/lib/python3.5/site-packages/pythonnC
RUN python -c "import pythonnC" 



RUN apt-get update \
      && apt-get install -y sudo \
      && rm -rf /var/lib/apt/lists/*
RUN echo "jovyan ALL=NOPASSWD: ALL" >> /etc/sudoers


RUN chown -R jovyan $HOME

USER $NB_USER

#The following are convenience aliases
#once inside the image make it such that a notebook can be created using 
#the dockerimage python, but the hosts browser and the hosts mounted file system.
RUN echo 'alias nb="jupyter-notebook --ip=* --no-browser"' >> ~/.bashrc
RUN echo 'alias mnt="cd /home/mnt"' >> ~/.bashrc
RUN echo 'alias erc="emacs ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias src="source ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias egg="cd /opt/conda/lib/python3.5/site-packages/"' >> ~/.bashrc 
RUN echo 'alias nu="cd /home/jovyan/work/scidash/neuronunit"' >> ~/.bashrc
#Note the line below is required in order for jNeuroML to work inside pyNeuroML.
ENV NEURON_HOME "/home/jovyan/neuron/nrn-7.4/x86_64" #This line is not effective so the 
#next line is a hack, that achieves the same objectives as those embodied in the command above:
RUN echo 'export NEURON_HOME=/home/jovyan/neuron/nrn-7.4/x86_64' >> ~/.bashrc
RUN echo 'alias model="cd /work/scidash/neuronunit/neuronunit/models"' >> ~/.bashrc
RUN echo 'alias sciunit="cd /work/scidash/sciunit"' >> ~/.bashrc
RUN echo 'alias nu="python -c "from neuronunit.models.reduced import ReducedModel""'
#RUN echo "export DJANGO_SETTINGS_MODULE=myproject.settings.production"> ~/.bashrc
#RUN echo "export DJANGO_SETTINGS_MODULE=nirla.settings"> ~/.bashrc
#RUN echo "heroku config:set DJANGO_SETTINGS_MODULE=nirla.settings --account personal"> ~/.bashrc

#I prefer to have password less sudo since it permits me 
#to quickly and easily modify the system interactively post dockerbuild.

WORKDIR /home/mnt
RUN pip install execnet

#WORKDIR /home/jovyan/mnt/sciunitopt
#ENTRYPOINT python -i /home/jovyan/mnt/sciunitopt/AIBS.py 






