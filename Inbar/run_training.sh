#!/bin/bash
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time 02:00:00
#SBATCH --account m3246
#SBATCH -J train-pm
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/2.0.1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
outdir=/global/homes/i/inbarsav/SymLoss/Inbar/storage
plotsoutdir=/global/homes/i/inbarsav/SymLoss/Inbar/plots
trainer=/global/homes/i/inbarsav/SymLoss/Inbar/train_regress_NN.py
json=/global/homes/i/inbarsav/SymLoss/Inbar/config_run.json
saveplots="True"
savenet="True"
# Run the training
#srun -l -u 
python3 $trainer --jsonfile $json --saveplots $saveplots --savenet $savenet --outdir $outdir --plotsoutdir $plotsoutdir
#--rank-gpu --ranks-per-node=${SLURM_NTASKS_PER_NODE} $@

