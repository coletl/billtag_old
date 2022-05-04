#!/bin/bash
#SBATCH --job-name=plda
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=FAIL,END
#SBATCH --output=code/analysis/plda_crs.log
#SBATCH --partition=andyhall

ulimit -s unlimited

cd /home/groups/jgrimmer/billtag

# load modules

ml python/3.9.0
source renv/python/virtualenvs/renv-python-3.9.0/bin/activate

python code/analysis/plda_crs.py
python code/analysis/plda_cbp.py

