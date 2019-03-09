
# move to a docker install script.
sudo /opt/conda/bin/pip install git+https://github.com/scidash/sciunit.git@dev --process-dependency-links --upgrade
sudo /opt/conda/bin/conda install -y cython

# sudo /opt/conda/bin/pip install git+https://github.com/brian-team/brian2.git
# faster glif
#sudo /opt/conda/bin/pip install git+https://github.com/russelljjarvis/AllenSDK.git
sudo /opt/conda/bin/conda remove -y bokeh
sudo /opt/conda/bin/pip install pyNN lazyarray
#  PyGnuplot

#sudo /opt/conda/bin/conda install -c benschneider pygnuplot

#sudo apt-get install -y gnuplot

sudo /opt/conda/bin/conda install -y numpy
sudo /opt/conda/bin/pip install allensdk --upgrade
