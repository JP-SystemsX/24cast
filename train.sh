#!/bin/bash

#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH -J test_run
#SBATCH -t 1-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reachi$
#SBATCH -p ai,tnt                                              # TODO check for your clusters partition
#SBATCH --output test_run_%A.out
#SBATCH --error test_run_%A.err
#SBATCH --array=2016-2017

module load Miniconda3
module load CUDA

conda activate /bigwork/nhwpbecj/.conda/envs/24cast
which python

export CUDA_VISIBLE_DEVICES=0

python run.py --year=$SLURM_ARRAY_TASK_ID