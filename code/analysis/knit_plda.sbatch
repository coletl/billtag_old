#!/bin/bash
#SBATCH --job-name=plda_crs
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=FAIL,END
#SBATCH --output=code/analysis/plda_crs.log
#SBATCH --partition=andyhall

ulimit -s unlimited

cd /home/groups/jgrimmer/billtag

# load modules

ml R/4.1.2 system pandoc gsl 
ml python3.9

Rscript -e "rmarkdown::render('code/analysis/plda_crs.Rmd')"
