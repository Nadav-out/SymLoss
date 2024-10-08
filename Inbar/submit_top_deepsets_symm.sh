#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time 140
#SBATCH -q regular
#SBATCH --account m3246
#SBATCH -J train-top_symm_%x_%j
#SBATCH -o logs/%x-%j.out
##SBATCH --array=0-4
#SBATCH --ntasks-per-node=4 


module load parallel

srun -l -u /global/homes/i/inbarsav/SymLoss/Inbar/run_training_top_symm.sh 

#srun -l -u parallel --jobs 5 /global/homes/i/inbarsav/SymLoss/Inbar/run_training_seed_broken_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_broken_1_e_3.json

# srun parallel --jobs 5 /global/homes/i/inbarsav/SymLoss/Inbar/run_training_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_equiv.json

# srun parallel --jobs 5 /global/homes/i/inbarsav/SymLoss/Inbar/run_training_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_equiv_skip.json

