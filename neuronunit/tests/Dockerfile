#author Russell Jarvis rjjarvis@asu.edu
#author Rick Gerkin rgerkin@asu.edu
FROM scidash/neuronunit
USER root


RUN pip install --upgrade pip
RUN pip install matplotlib

RUN easy_install gevent
RUN easy_install greenlet
RUN pip install git+https://github.com/rkern/line_profiler

#RUN python -m ipykernel install
#RUN python setup.py install

ENV ATOM_VERSION v1.12.7
RUN apt-get update
#RUN apt-get install -y gedit meld gvfs-bin iceweasel
RUN apt-get install -y curl apt-utils

#RUN curl -L https://github.com/atom/atom/releases/download/${ATOM_VERSION}/atom-amd64.deb > /tmp/atom.deb && dpkg -i /tmp/atom.deb && rm -f /tmp/atom.deb

#Install a browser:
#RUN apt-get update && apt-get install -y xvfb chromium
#RUN ln -s /usr/bin/xvfb-chromium /usr/bin/google-chrome
#RUN ln -s /usr/bin/xvfb-chromium /usr/bin/chromium-browse


ENV WORK_HOME $HOME/work/scidash
RUN ipython profile create default
RUN apt-get install -y libxss1
RUN git clone https://github.com/soravux/scoop
WORKDIR scoop
RUN echo "got here"
RUN python setup.py install
RUN apt-get install -y python3-setuptools
ENV PATH $PATH:/opt/conda/bin
RUN easy_install gevent
RUN easy_install greenlet
RUN apt-get install -y povray eog
WORKDIR $HOME
RUN git clone https://github.com/russelljjarvis/BluePyOpt.git
WORKDIR BluePyOpt
RUN python setup.py install
RUN apt-get install -y python-tk
RUN chown -R jovyan $HOME
RUN python -c "import bluepyopt"
RUN conda install mpi4py
RUN pip install git+https://github.com/roryk/ipython-cluster-helper

RUN ipcluster nbextension enable
#RUN chmod 755 jovyan/ipython/profile_default
#RUN chmod 744 jovyan/ipython/profile_default

RUN chown -R jovyan /opt/conda/lib/python3.5/site-packages/

USER $NB_USER
#RUN /bin/bash -c "source /etc/sudoers"
#RUN sudo ipcluster nbextension enable
#RUN sudo chmod 755 jovyan/ipython/profile_default
#RUN sudo chmod 744 jovyan/ipython/profile_default

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
RUN echo 'blah'
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


#The following are convenience aliases
RUN echo 'alias pynml=pynml-povray NML2_MultiCompCellNetwork.nml -scalez 15 -conns -conn_points'
RUN echo 'alias nml2= povray NML2_MultiCompCellNetwork.pov'

RUN echo 'alias nb="jupyter notebook --browswer iceweasel"' >> ~/.bashrc
RUN echo 'alias st="cd /home/mnt/neuronunit/software_tests"' >> ~/.bashrc
RUN echo 'alias mnt="cd /home/mnt"' >> ~/.bashrc
RUN echo 'alias erc="emacs ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias src="source ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias egg="cd /opt/conda/lib/python3.5/site-packages/"' >> ~/.bashrc
RUN echo 'alias nu="cd /home/jovyan/work/scidash/neuronunit"' >> ~/.bashrc
RUN echo 'alias model="cd /work/scidash/neuronunit/neuronunit/models"' >> ~/.bashrc
RUN echo 'alias sciunit="cd /work/scidash/sciunit"' >> ~/.bashrc
RUN echo 'alias nu="python -c "from neuronunit.models.reduced import ReducedModel""'
RUN export PATH=$PATH

#un comment the following line to test ipcluster during build.
#RUN ipcluster start --profile=chase --debug &
#https://github.com/russelljjarvis/neuronunit.git
WORKDIR $HOME
RUN echo "do this part again such that pickled files can be used to bypass problems with aibs.py in the short term fdsa"
RUN echo "do this part again such that pickled files can be used to bypass problems with aibs.py in the short term"
RUN echo "definitely not a long term fix"
RUN git clone -b dev https://github.com/russelljjarvis/neuronunit.git
WORKDIR $HOME/neuronunit/neuronunit/tests/NeuroML2

RUN nrnivmodl
RUN ls LEMS_2007One.xml
WORKDIR $HOME/neuronunit/neuronunit/tests/
#ENTRYPOINT python -m scoop exhaustive_search.py
RUN ls *.pickle
ENTRYPOINT python -m scoop nsga.py && python -m scoop exhaustive_search.py
