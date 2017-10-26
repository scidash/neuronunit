# neuronunit
# author Rick Gerkin rgerkin@asu.edu

FROM scidash/neuron-mpi-neuroml

USER root
ARG MOD_DATE=0
RUN echo $MOD_DATE
ADD . $HOME/neuronunit
RUN chown -R $NB_USER $HOME 
WORKDIR $HOME/neuronunit 

# Install neuronunit and dependencies.
RUN python setup.py install

# Run all unit tests.
ENTRYPOINT python -m unittest unit_test/core_tests.py
