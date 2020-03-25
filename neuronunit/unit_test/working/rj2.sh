#!/bin/bash
#SBATCH -n 20                        # number of cores
#SBATCH -t 0-90:40                  # wall time (D-HH:MM)
#SBATCH -A rjjarvis                 # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o nuopt.%j.out             # STDOUT (%j = JobId)
#SBATCH -e nuopt.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=rjjarvis@asu.edu # send-to address
#rm /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/nuopt*.err
#rm /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/nuopt*.out

#python /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/opt_all_e.py
#python /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/simulated_data_old.py

##SBATCH --nodes=2
#python /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/sim_data_long_run.py
#python /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/real_data_long_run.py
#python /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/sim_data_unit_test.py
#python 
echo sim_data_long_run.py real_data_long_run.py | xargs python | xargs parallel

python real_data_long_run.py
python sim_data_long_run.py
python /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/mill_books_data_driven.py
python /home/rjjarvis/NeuronunitOpt/neuronunit/unit_test/working/mill_books_sim_data.py
