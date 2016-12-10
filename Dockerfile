FROM scidash/neuronunit-docs

ADD . /home/mnt
WORKDIR /home/mnt
RUN pip install nbconvert ipykernel
USER root
RUN chmod -R 777 .
USER $NB_USER
ENTRYPOINT jupyter nbconvert --to notebook --execute docs/chapter1.ipynb
