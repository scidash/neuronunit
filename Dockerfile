# neuronunit
# author Rick Gerkin rgerkin@asu.edu

FROM scidash/neuron-mpi-neuroml

USER root
RUN chown -R $NB_USER . 
USER $NB_USER

# Make neuronunit source directory in Travis image visible to Docker.  
ADD . . 

# Install neuronunit and dependencies.
RUN python setup.py install

# Run all unit tests.
ENTRYPOINT python -m unittest test_*.py

WORKDIR $WORKDIR