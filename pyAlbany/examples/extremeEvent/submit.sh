#!/bin/bash
# Submission script for Blake
#SBATCH --job-name=extremeEvent
#SBATCH --time=41:00:00 # hh:mm:ss
#
#SBATCH -N 1
#SBATCH -p blake

python thermal_steady.py &> log_steady.txt
