apt-get install -y cpp gcc
apt-get install -y libx11-6 python-dev git build-essential
apt-get install -y autoconf automake gcc g++ make gfortran
apt-get install -y python-tables
apt-get install -y libhdf5-serial-dev
conda install numpy;
conda install numba;
conda install dask;
pip install pip --upgrade;
pip install tables
pip install scipy==1.5.4
pip install -e .
pip install coverage
git clone -b neuronunit https://github.com/russelljjarvis/jit_hub.git
cd jit_hub; pip install -e .; cd ..;
pip install cython
pip install asciiplotlib;
git clone -b master https://github.com/russelljjarvis/BluePyOpt.git
cd BluePyOpt; pip install -e .
pip install git+https://github.com/russelljjarvis/eFEL
pip install ipfx
pip install streamlit
pip install sklearn
pip install seaborn
pip install frozendict
pip install plotly
pip install --upgrade colorama
rm -rf /opt/conda/lib/python3.8/site-packages/sciunit
git clone -b dev https://github.com/russelljjarvis/sciunit.git
cd sciunit; pip install -e .; cd ..;
pip install allensdk==0.16.3
