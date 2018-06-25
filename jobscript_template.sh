#!/bin/bash
#SBATCH --account=def-kristen
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=1024M
#SBATCH --job-name=SigmaRV_<<band_str>>_<<R>>_<<Teff>>_<<logg>>
python2.7 sigmaRV_grid.py <<band_str>> <<R>> <<Teff>> <<logg>>
