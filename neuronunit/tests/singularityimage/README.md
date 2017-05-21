# Command Line Access to Comet (of Neuro Science Gateway)
## Quick start guide:

To prepare singularity images for NSG-HPC see steps below:

## Copy relevant files with scp 

Use to Comet using the secure copy (secure copy) protocal.

`scp nu.def rjjarvis@comet.sdsc.xsede.org:~/`

## Use rsync for bigger files

copy bigger files as it is better at recovery from interruption.

`rsync pnp_latest-2017-05-19-f1d9712ba440.img rjjarvis@comet.sdsc.xsede.org:~/`

## Create a test SLURM script with contents like the following:
```
#!/bin/bash
#SBATCH --job-name=test_singularity
#SBATCH --time 05:00
##SBATCH -t 0-2:00 # time (D-HH:MM) 
#SBATCH --nodes 1
#SBATCH --output=singularity_test.txt
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH -o slurm.%N.%j.out # STDOUT 
#SBATCH -e slurm.%N.%j.err # STDERR s
#SBATCH --mail-type=END,FAIL 
# notifications for job done & fail 
#SBATCH --mail-user=rjjarvis@asu.edu # send-to address  
##SBATCH --mem-per-cpu=500


module load singularity
singularity bootstrap pnp_latest-2017-05-19-f1d9712ba440.img nu.def
```
# Submit the SLURM script to the queue
If the files name is launch_singularity.sh launch the file with:
`
sbatch launch_singularity.sh
`
# The graphical way.

Some considerations for people who are already used to running code on clusters. NSG removes the need to write an explicit PBS script or a SLURM scheduling script, although you will be prompted to enter some of the information contained in these scripts in step 4. You will also not be required to write an $openmpi or $mpiexec launch command, such a command will be constructed for you based on the information you provide.

## Step 1
Under data simply upload a compressed folder that contains all of your project content, you will be asked to specify the main simulation launching file later, if it is not guessed automatically. NSG understands both *.tar.bz and *.zip.

## Step 2.
Under tasks select create a new task.

## Step 3.
Under tools select NEURON7.4 Python on Comet (7.4) - Using Python to run NEURON 7.
The help desk is very responsive, and they can add python modules as needed. They usually respond by the end of the day.

## Step 4.
Input parameters is where you select number of hosts, number of CPU cores time to run, and the Python file to launch. In my example that file is init.py, if you use that convention of calling the main file init.py or init.hoc NSG will probably guess that file for you. Input parameters is probably the graphical equivalent of a PBS or SLURM script.

## Step 5.
Then press save and run.

## Step 6.
You can check job progress and stdout.txt while the job is executing by pressing Tasks and navigating to view status, or view output as appropriate.

# Prepare singularity images (.img) for NSG-HPC:

On OSX sinularity can easily be installed by creating a dedicated Vagrant Ubuntu:latest image, and installing singularity within in it

# 1 On linux build singularity from source:
```
$ mkdir ~/git
$ cd ~/git
$ git clone https://github.com/singularityware/singularity.git
$ cd singularity
$ ./autogen.sh
```
Autogen did not work, even after supplying automake as described here, and using `sudo apt-get install automake`
https://geeksww.com/tutorials/libraries/m4/installation/installing_m4_macro_processor_ubuntu_linux.php
Installing the yum packages specified below
https://hpc.nih.gov/apps/singularity.html
in apt-get as opposed to yum would probably work, but no time. Skip to install approach # 2.
```
$ ./configure --prefix=/usr/local --sysconfdir=/etc
$ make
$ sudo make install
```
# 2 Workaround:
```
VERSION=2.2.1
wget https://github.com/singularityware/singularity/releases/download/$VERSION/singularity-$VERSION.tar.gz
tar xvf singularity-$VERSION.tar.gz
cd singularity-$VERSION
./configure --prefix=/usr/local
make
sudo make install
```

# 3 Convert the local docker container to a local singularity image
Alias to build a singularity image based on a local docker container:
```
alias d2s='sudo docker run -v /var/run/docker.sock:/var/run/docker.sock -v `pwd`/singularity\image:/output --privileged -t --rm singularityware/docker2singularity pnp:latest'


```

This results in a directory 
```
`pwd`/singularityimage/
```
containing the file:
```
pnp_latest-2017-04-10-e61582246138.img
```
# 4 Enter the Sinularity image
Then to enter the singularity vm (as opposed to docker container):
```
sudo singularity shell -w pnp_latest-2017-04-10-e61582246138.img 
Singularity: Invoking an interactive shell within container...

root@rjjarvis:/root# su jovyan
jovyan@rjjarvis:/root$ cd ~
jovyan@rjjarvis:~$ ls
```
Singularity image is build from local container, by entering a different special purpose singularity written docker container, whose only only function is to write a singularity image, based on the supplied docker container: 

Giving us an interactive session in our familiar docker like environment:
```
jovyan@rjjarvis:~$ ls
jLEMS     LEMS        NeuroML2    nrn-7.4        org.neuroml1.model  org.neuroml.import  org.neuroml.model.injectingplugin  sciunit  x86_64
jNeuroML  libNeuroML  neuronunit  openmpi-2.0.0  org.neuroml.export  org.neuroml.model   pylems                             work
```

# 5 Try launching a job interactively within the singularity image

if jobs can not be launched without modification to the image, collate all of the BASH commands in a definitions file.

(.def), this file can be launched upon invocation of singularity via the `bootstrap` singularity invocation. Above you will notice that the NSG-HPC slurm script invokes singularity via this `bootstrap` pattern with the bottom two lines: 
```
module load singularity
singularity bootstrap pnp_latest-2017-05-19-f1d9712ba440.img nu.def
```
These definition files serve as an analogous role to Dockeriles with the difference that that the def file is much closer to pure bash, and the def file does not build the image, the image is already built, and the def file is used to initialise the image after its launched for usages where you don't intend to enter the image interactively, you just want singularity to run through a process of automated steps. 

For an example of definition file contents see: [nu.def](https://github.com/russelljjarvis/neuronunit/edit/dev/neuronunit/tests/singularityimage/README.md)




