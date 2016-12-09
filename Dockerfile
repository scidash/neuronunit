FROM scidash/neuronunit-docs

ADD . /home/mnt
WORKDIR /home/mnt
RUN pip install nbconvert ipykernel
CMD ls
CMD pwd
ENTRYPOINT ["ls"]
# jupyter nbconvert --to notebook --execute docs/chapter1.ipynb
