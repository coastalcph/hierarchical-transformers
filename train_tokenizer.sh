#!/bin/bash
#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH -p image1 --mem=32GB
#SBATCH --time=60:00:00
#SBATCH --output=tokenizer.txt
#SBATCH --job-name=tokenizer

hostname
export PYTHONPATH=.
python language_modelling/train_tokenizer.py