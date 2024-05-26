#!/bin/bash -l
#SBATCH --chdir /scratch/izar/awicht
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --time 06:00:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559

cd /home/awicht/deepl
source /home/awicht/venv/bin/activate
python /home/awicht/deepl/MEGA_confusion_custom_loss.py
echo "finished"
