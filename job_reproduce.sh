#!/bin/bash
# run_train.sh

TEMP_END=${1:-0.825}

# Set up micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate gflow

module load eth_proxy

# Set Hugging Face cache directory
export HUGGINGFACE_HUB_CACHE=/cluster/work/cotterell/mdeprada/hf-cache

# Run training script
python -u replicate_experiment.py