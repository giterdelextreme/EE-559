#!/bin/bash -l
#SBATCH --chdir /scratch/izar/awicht
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --time 00:00:30
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559

cd /home/awicht/deepl
source /home/awicht/venv/bin/activate
python /home/awicht/test_import/test.py
echo "finished"