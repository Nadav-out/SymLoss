#!/bin/bash

module load parallel


for ((i = 0 ; i < 6 ; i++ )); do sbatch -C gpu -q regular  -t 140 --nodes 1 --ntasks-per-node=1 --gpus-per-node=1  -A m3246 -J 1e3_$i /global/homes/i/inbarsav/SymLoss/Inbar/run_training_seed_broken_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_broken_1_e_3.json ; done 

for ((i = 0 ; i < 6 ; i++ )); do sbatch -C gpu -q regular  -t 140 --nodes 1 --ntasks-per-node=1 --gpus-per-node=1  -A m3246 -J 1e1_$i /global/homes/i/inbarsav/SymLoss/Inbar/run_training_seed_broken_for_batch.sh /global/homes/i/inbarsav/SymLoss/Inbar/config_run_broken_1_e_1.json ; done  

