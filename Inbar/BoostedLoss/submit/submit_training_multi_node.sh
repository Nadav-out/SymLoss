#!/bin/bash

# Job name
#SBATCH --job-name=Speed_MSE

# choose the GPU queue
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --account=m3246_g
#SBATCH --array=0-4

#requesting one node
#SBATCH -N 2                          # number of nodes you want to use
#SBATCH --ntasks-per-node=4                     # number of processes to be run
#SBATCH --gpus-per-node=4             # every process wants one GPU!
#SBATCH --image=nersc/pytorch:ngc-23.07-v0
#SBATCH --module=gpu,nccl-2.18


##SBATCH --time=07:00:00
#SBATCH --time=10:00:00

# mail on failures
#SBATCH --mail-user=thandi.madula.17@ucl.ac.uk
#SBATCH --mail-type=ALL

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/global/homes/t/tmadula/Symmetries/submit/logs/slurm-%j.%x.out

# OpenMP settings:
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread


cd /global/homes/t/tmadula/BoostedLoss/

echo "Moved dir, now in:"
pwd

DATADIR=/pscratch/sd/t/tmadula/data/ATLASTOP/
# echo "Activating environment"
# module load pytorch/2.0.1

 
echo $CONDA_DEFAULT_ENV

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export CUDA_VISIBLE_DEVICES=3,2,1,0

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

echo "Job array ID:"
echo $SLURM_ARRAY_TASK_ID



# Training
srun -u shifter -V ${DATADIR}:/ATLASTOP -V /global/homes/t/tmadula/BoostedLoss/:/submit \
bash -c "
    source /global/homes/t/tmadula/BoostedLoss/submit/export_DDP_vars.sh
    python -u train_transformer_top_multinode.py --data_dir /pscratch/sd/t/tmadula/data/ATLASTOP/ --output_dir /pscratch/sd/t/tmadula/data/ATLASTOP/results/neurips/ --save_tag $SLURM_ARRAY_TASK_ID --boost_type 3D --apply_penalty
    "

# Evaluation
# srun -u shifter -V ${DATADIR}:/ATLASTOP -V /global/homes/t/tmadula/BoostedLoss/:/submit \
# bash -c "
#     source /global/homes/t/tmadula/BoostedLoss/submit/export_DDP_vars.sh
#     python -u Evaluate.py
#     "