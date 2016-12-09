FROM scidash/neuronunit-docs

ADD . /home/mnt
WORKDIR /home/mnt
RUN pip install nbconvert
ENTRYPOINT ["jupyter nbconvert --to notebook --execute docs/chapter1.ipynb"]