docker build scipy-notebook-plus -t scidash/scipy-notebook-plus
docker build neuron-mpi-neuroml -t scidash/neuron-mpi-neuroml
docker build neuronunit -t scidash/neuronunit

for stack in scipy-notebook-plus neuron-mpi-neuroml neuronunit neuronunit-docs; do
    docker build $stack -t scidash/$stack
done