## Command Line Access to Comet (of Neuro Science Gateway)
# Quick start guide:

# Copy relevant files with scp 

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
##
## The graphical way.

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
