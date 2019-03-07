#!/bin/bash
sudo /home/russell/anaconda3/bin/conda install -y cython
sudo /home/russell/anaconda3/bin/conda install -y numpy deap dask numba
# move to a docker install script.
sudo pip install git+https://github.com/scidash/sciunit.git@dev --upgrade #--process-dependency-links --upgrade

sudo pip3 install git+https://github.com/brian-team/brian2.git
# faster glif
sudo pip3 install git+https://github.com/russelljjarvis/AllenSDK.git
sudo pip3 install allensdk --upgrade
sudo pip3 install -e . --ignore-installed


#sudo /opt/conda/bin/conda remove -y bokeh
#sudo /opt/conda/bin/pip install pyNN lazyarray
#  PyGnuplot

#sudo /opt/conda/bin/conda install -c benschneider pygnuplot

#sudo apt-get install -y gnuplot

