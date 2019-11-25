FROM jupyter/scipy-notebook
USER root

RUN chown -R $NB_USER $HOME

#Get a whole lot of GNU core development tools
#version control java development, maven
#Libraries required for building MPI from source
#Libraries required for building NEURON from source

RUN apt-get update
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
RUN bash miniconda.sh -b -p $HOME/miniconda
RUN export PATH="$HOME/miniconda/bin:$PATH"
# RUN apt-get install python-lxml
# RUN apt-get install python-lxml

RUN hash -r
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda info -a
RUN pip install -U pip
RUN pip install --upgrade pip
RUN pip install coveralls
RUN pip install git+https://github.com/fun-zoological-computing/AllenSDK
RUN pip install pyneuroml
RUN pip install git+https://github.com/NeuralEnsemble/libNeuroML
RUN pip install git+https://github.com/scidash/sciunit@dev
RUN pip install dask distributed brian2 numba
RUN pip install --upgrade beautifulsoup4
RUN pip install neurodynex
RUN pip install pyneuroml
RUN pip install sklearn
RUN pip install seaborn
RUN pip install dask-ml # dask machine learning
ADD . .
#RUN pip install git+https://github.com/russelljjarvis/neuronunit.git
# WORKDIR neuronunit

RUN pip install -r requirements.txt
RUN pip install .

#script:
RUN sudo /opt/conda/bin/pip install git+https://github.com/plotly/dash
RUN git clone https://github.com/hfwittmann/dash.git


RUN sudo /opt/conda/bin/pip install -r dash/dash-simple/requirements-simple.txt
WORKDIR $HOME
EXPOSE 80

WORKDIR neuronunit/unit_test/working
# RUN jupyter nbconvert --execute examples/chapter3.ipynb
# RUN jupyter nbconvert --execute examples/example_data.ipynb

ENTRYPOINT ["python", "examples/use_edt.py"]
