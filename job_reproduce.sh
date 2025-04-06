#!/bin/bash
#SBATCH --output=job_reproduce-%j.out

TEMP_END=${1:-0.825}
source ~/.bashrc
# Set up micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate gflow

module load eth_proxy

# Set Hugging Face cache directory
export HUGGINGFACE_HUB_CACHE=/cluster/work/cotterell/mdeprada/hf-cache

# Run training script
python -u replicate_experiment.py hf