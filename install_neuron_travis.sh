#!/bin/bash
sudo apt install libx11-dev git bison flex automake libtool libxext-dev libncurses-dev python3-dev xfonts-100dpi cython3 libopenmpi-dev python3-scipy make zlib1g-dev
 git clone https://github.com/neuronsimulator/nrn
 cd nrn
 ./configure --with-nrnpython=python3 --without-iv --without-paranrn
 ./build.sh
 make -j
 sudo make install -j
 cd src/nrnpython
 sudo python3 setup.py install
 #1848  cd /usr/local/nrn/share/nrn/lib/python
 #1909  cd ~/git/nrn/
 #1919  cd nrnpython/
 #1923  nrniv
 #1924  which nrniv
 #1925  cd ~/git/nrn
 #1945  ./configure --with-iv --with-paranrn --with-nrnpython=python3
 #1946  ./configure --without-iv --without-paranrn --with-nrnpython=python3
 #8212  ./configure --with-nrnpython=python3\nmake -j\nsudo make install -j\ncd src/nrnpython\nsudo python3 setup.py install
 #8214  ./configure --with-iv --with-paranrn --with-nrnpython=python3
 #8216  ./configure --without-iv --without-paranrn --with-nrnpython=python3
 #8217  make -j\nsudo make install -j\ncd src/nrnpython\nsudo python3 setup.py install
