FROM scidash/neuronunit-optimization
USER jovyan
RUN pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN pip install dask
RUN pip install distributed
RUN sudo apt-get update
RUN pip install ioloop
RUN sudo chown -R jovyan /home/jovyan
RUN pip install git+https://github.com/OpenSourceBrain/OpenCortex
RUN git clone https://github.com/vrhaynes/AllenInstituteNeuroML.git
RUN pip install PyLEMS

# RUN sudo /opt/conda/bin/pip install git+https://github.com/python-quantities/python-quantities
# RUN sudo /opt/conda/bin/pip install git+https://github.com/scidash/sciunit@dev
RUN sudo chown -R jovyan /home/jovyan
WORKDIR /home/jovyan/neuronunit/neuronunit/unit_test
RUN sudo chown -R jovyan /home/jovyan
RUN git clone https://github.com/vrhaynes/AllenInstituteNeuroML.git
RUN pip install git+https://github.com/OpenSourceBrain/OpenCortex
# RUN sudo apt-get -y install ipython ipython-notebook
# RUN sudo -H /opt/conda/bin/pip install jupyter
# ADD neuronunit/unit_test/post_install.sh .

RUN git clone https://github.com/OpenSourceBrain/osb-model-validation.git
WORKDIR osb-model-validation
RUN python setup.py install 
RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow 

RUN sudo /opt/conda/bin/python3 -m ipykernel.kernelspec

# Then install the Jupyter Notebook using:
RUN pip install jupyter

RUN sudo /opt/conda/bin/pip uninstall -y tornado
RUN pip install tornado==4.5.3
RUN /opt/conda/bin/python3 -m pip install ipykernel
RUN /opt/conda/bin/python3 -m ipykernel install --user
RUN pip install deap
WORKDIR $HOME
# ADD . neuronunit
# WORKDIR neuronunit
# RUN sudo /opt/conda/bin/pip install -e .
#RUN bash post_install.sh
ENTRYPOINT /bin/bash

