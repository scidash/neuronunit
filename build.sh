apt-get install -y cpp gcc
apt-get install -y libx11-6 python-dev git build-essential
apt-get install -y autoconf automake gcc g++ make gfortran
apt-get install -y python-tables
apt-get install -y libhdf5-serial-dev
/home/rudolph/anaconda3/bin/conda install numpy;
/home/rudolph/anaconda3/bin/conda install numba;
/home/rudolph/anaconda3/bin/conda install dask;
/home/rudolph/anaconda3/bin/pip install pip --upgrade;
/home/rudolph/anaconda3/bin/pip install tables
/home/rudolph/anaconda3/bin/pip install scipy==1.5.4
/home/rudolph/anaconda3/bin/pip install -e .
/home/rudolph/anaconda3/bin/pip install coverage
git clone -b neuronunit https://github.com/russelljjarvis/jit_hub.git
cd jit_hub; pip install -e .; cd ..;
/home/rudolph/anaconda3/bin/pip install cython
/home/rudolph/anaconda3/bin/pip install asciiplotlib;
git clone -b master https://github.com/russelljjarvis/BluePyOpt.git
cd BluePyOpt; pip install -e .
/home/rudolph/anaconda3/bin/pip install git+https://github.com/russelljjarvis/eFEL
/home/rudolph/anaconda3/bin/pip install ipfx
/home/rudolph/anaconda3/bin/pip install streamlit
/home/rudolph/anaconda3/bin/pip install sklearn
/home/rudolph/anaconda3/bin/pip install seaborn
/home/rudolph/anaconda3/bin/pip install frozendict
/home/rudolph/anaconda3/bin/pip install plotly
/home/rudolph/anaconda3/bin/pip install --upgrade colorama
rm -rf /opt/conda/lib/python3.8/site-packages/sciunit
git clone -b dev https://github.com/russelljjarvis/sciunit.git
cd sciunit; pip install -e .; cd ..;
/home/rudolph/anaconda3/bin/pip install allensdk==0.16.3

/home/rudolph/anaconda3/bin/pip install pycuda
