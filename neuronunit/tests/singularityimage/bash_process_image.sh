#!/bin/bash

#delete old images if they exist
if [ -f "*.img" ] ; then rm *.img;  fi
#derive a singularity image from translating a docker container
sudo docker run -v /var/run/docker.sock:/var/run/docker.sock -v `pwd`/singularity\image:/output --privileged -t --rm singularityware/docker2singularity pnp:latest
#bootstrap some bash commands into the image, they effects should be lasting because VM images can have persistance.
sudo singularity bootstrap pnp*.img nu.def
#Add the required files.
sudo singularity exec pnp*.img python 

sudo rsync pnp*.img rjjarvis@comet.sdsc.xsede.org:~/
#boot strap file no longer required, as bootstrapping is done locally.
#sudo scp nu.def rjjarvis@comet.sdsc.xsede.org:~/
