# neuronunit
# author Rick Gerkin rgerkin@asu.edu

FROM scidash/neuron-mpi-neuroml

USER root
RUN chown -R $NB_USER . 
USER $NB_USER

# Make neuronunit source directory in Travis image visible to Docker.  
ADD . $HOME/neuronunit
WORKDIR $HOME/neuronunit 

# Install neuronunit and dependencies.
RUN pip install .

# Run all unit tests.
ENTRYPOINT python -m unittest unit_test/test_*.py