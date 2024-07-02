#!/bin/bash
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time 02:00:00
#SBATCH -J train-pm
#SBATCH -o logs/%x-%j.out
#SBATCH --qos=regular
#SBATCH --nodes=1


module load parallel

srun parallel --jobs 5 /global/homes/i/inbarsav/SymLoss/Inbar/run_training_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_SGD.json

srun parallel --jobs 5 /global/homes/i/inbarsav/SymLoss/Inbar/run_training_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_equiv.json

srun parallel --jobs 5 /global/homes/i/inbarsav/SymLoss/Inbar/run_training_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_equiv_skip.json