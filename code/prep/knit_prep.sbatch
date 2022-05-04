#!/bin/bash
#SBATCH --job-name=legis_prep
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL,END
#SBATCH --output=code/prep/knit_prep.log
#SBATCH --partition=andyhall

ulimit -s unlimited

cd /home/groups/jgrimmer/billtag

# load modules

ml R/4.1.2 system pandoc gsl 
Rscript -e "rmarkdown::render('code/prep/0-get_govinfo.Rmd')"
Rscript -e "rmarkdown::render('code/prep/0-prep_labels.Rmd')"

ml python/3.9.0
source renv/python/virtualenvs/renv-python-3.9.0/bin/activate
python code/prep/1-corpus_crs.py
python code/prep/1-corpus_crs.py
