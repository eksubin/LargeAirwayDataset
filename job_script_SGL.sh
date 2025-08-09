#!/bin/bash
#$ -N ThreeDDavid         # Name of the job
#$ -q SGL                   # Queue name
#$ -pe smp 16               # Number of slots (cores)
#$ -l ngpus=1                 # Number of GPUs
#$ -cwd                     # Use the current working directory
#$ -o ./logs/VanillaUNet3D/output.log            # Output log file
#$ -e ./logs/VanillaUNet3D/error.log             # Error log file

conda activate unet3D
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2
python VanillaUNet3DFineTuneDavid.py