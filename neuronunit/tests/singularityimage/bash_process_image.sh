#!/bin/bash

if [ -f "*.img" ] ; then rm *.img;  fi

sudo docker run -v /var/run/docker.sock:/var/run/docker.sock -v `pwd`/singularity\image:/output --privileged -t --rm singularityware/docker2singularity pnp:latest
sudo singularity bootstrap pnp_latest-2017-05-19-f1d9712ba440.img nu.def

