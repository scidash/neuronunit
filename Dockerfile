FROM russelljarvis/neuronunit
# FROM scidash/scipy-notebook-plus
USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN pip install dask
RUN pip install distributed
RUN pip install tornado deap 
RUN sudo apt-get update
# RUN sudo apt-get install -y graphviz
# RUN conda install -y bokeh -c bokeh
RUN pip install pyzmq
# RUN pip install graphviz
RUN sudo chown -R jovyan /home/jovyan
RUN sudo /opt/conda/bin/pip install git+https://github.com/OpenSourceBrain/OpenCortex
RUN git clone https://github.com/vrhaynes/AllenInstituteNeuroML.git
RUN sudo /opt/conda/bin/pip install PyLEMS
# RUN sudo /opt/conda/bin/pip install -e russell/neuronunit/neuronunit
# RUN /opt/conda/bin/pip install -e BluePyOpt
RUN sudo /opt/conda/bin/pip uninstall -y sciunit
# RUN sudo /opt/conda/bin/pip uninstall -y quantities
RUN sudo rm -r /opt/conda/lib/python3.5/site-packages/quantities-0.11.1-py3.5.egg
RUN sudo /opt/conda/bin/pip install git+https://github.com/python-quantities/python-quantities
RUN sudo /opt/conda/bin/pip install git+https://github.com/scidash/sciunit@dev
RUN sudo chown -R jovyan /home/jovyan
WORKDIR /home/jovyan/neuronunit/neuronunit/unit_test
RUN sudo chown -R jovyan /home/jovyan
RUN git clone https://github.com/vrhaynes/AllenInstituteNeuroML.git
RUN sudo /opt/conda/bin/pip install git+https://github.com/OpenSourceBrain/OpenCortex
RUN touch post_install.sh
RUN echo "sudo /opt/conda/bin/pip install -e BluePyOpt" >> post_install.sh
RUN echo "sudo /opt/conda/bin/pip install -e neuronunit" >> post_install.sh
RUN echo "sudo /opt/conda/bin/pip install -e git+https://github.com/NeuroML/pyNeuroML" >> post_install.sh
WORKDIR $HOME
RUN rm -r scoop
RUN rm -r work
RUN sudo /opt/conda/bin/conda update conda

#RUN sudo /bin/bash post_install.sh
RUN sudo apt-get -y install ipython ipython-notebook
RUN sudo -H /opt/conda/bin/pip install jupyter
RUN sudo /opt/conda/bin/pip install pyzmq --upgrade
# ENTRYPOINT /bin/bash
