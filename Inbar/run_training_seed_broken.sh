#!/bin/bash
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time 02:00:00
#SBATCH --account m3246
#SBATCH -J train-pm
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/2.0.1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
outdir=/pscratch/sd/i/inbarsav/SymmLoss/storage
plotsoutdir=/global/homes/i/inbarsav/SymLoss/Inbar/plots
trainer=/global/homes/i/inbarsav/SymLoss/Inbar/train_regress_NN_broken.py
json=/global/homes/i/inbarsav/SymLoss/Inbar/config_run_broken_0.json
saveplots="True"
savenet="True"
seeddata="rand"
seedtrain="rand"
# Run the training
#srun -l -u --rank-gpu --ranks-per-node=${SLURM_NTASKS_PER_NODE} $@
python3 $trainer --jsonfile $json --saveplots $saveplots --savenet $savenet --outdir $outdir --plotsoutdir $plotsoutdir --seeddata $seeddata --seedtrain $seedtrain


