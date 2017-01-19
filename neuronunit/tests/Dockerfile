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
RUN apt-get install -y gedit meld gvfs-bin iceweasel
RUN apt-get install -y curl



RUN curl -L https://github.com/atom/atom/releases/download/${ATOM_VERSION}/atom-amd64.deb > /tmp/atom.deb && dpkg -i /tmp/atom.deb && rm -f /tmp/atom.deb

#Install a browser:
RUN apt-get update && apt-get install -y xvfb chromium
#ADD xvfb-chromium /usr/bin/xvfb-chromium
RUN ln -s /usr/bin/xvfb-chromium /usr/bin/google-chrome
RUN ln -s /usr/bin/xvfb-chromium /usr/bin/chromium-browse



ENV WORK_HOME $HOME/work/scidash


RUN ipython profile create default
#
RUN apt-get install -y libxss1




RUN git clone https://github.com/soravux/scoop
WORKDIR scoop
RUN python setup.py install




RUN apt-get install -y python3-setuptools
ENV PATH $PATH:/opt/conda/bin


RUN easy_install gevent
RUN easy_install greenlet

RUN apt-get install -y povray eog
WORKDIR $HOME
RUN git clone https://github.com/BlueBrain/BluePyOpt

RUN pip install git+https://github.com/uqfoundation/multiprocess
RUN apt-get install -y python-tk

RUN chown -R jovyan $HOME

RUN pip install bluepyopt
RUN python -c "import bluepyopt"



RUN pip install git+https://github.com/roryk/ipython-cluster-helper

RUN ipcluster nbextension enable
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



#The following are convenience aliases
RUN echo 'alias pynml=pynml-povray NML2_MultiCompCellNetwork.nml -scalez 15 -conns -conn_points'
RUN echo 'alias nml2= povray NML2_MultiCompCellNetwork.pov'


RUN export PATH=$PATH

WORKDIR $HOME 
#RUN ipcluster start --profile=jovyan --debug &

WORKDIR $HOME
RUN git clone -b russell_dev https://github.com/scidash/neuronunit.git
WORKDIR $HOME/neuronunit/neuronunit/tests/NeuroML2

RUN nrnivmodl
RUN ls LEMS_2007One.xml
#WORKDIR $HOME/neuronunit/neuronunit/tests/
WORKDIR ../
RUN touch run_ipp.py
RUN echo "import os">> run_ipp1.py
RUN echo "os.system('ipcluster start --profile=jovyan --debug &')" >> run_ipp1.py

RUN echo "import ipyparallel as ipp ; rc = ipp.Client(profile='jovyan'); print('hello from before cpu '); print(rc.ids);" >> run_ipp2.py
#ENTRYPOINT cat run_ipp.py && ipython run_ipp1.py && sleep 15 && ipython run_ipp2.py 

