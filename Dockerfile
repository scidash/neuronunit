FROM scidash/neuron-mpi-neuroml

ADD . /home/mnt
WORKDIR /home/mnt
USER root
RUN chown -R $NB_USER . 
USER $NB_USER
RUN python setup.py install