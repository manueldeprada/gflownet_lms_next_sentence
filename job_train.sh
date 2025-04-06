#!/bin/bash
#SBATCH --output=job_train-%j.out

TEMP_END=${1:-0.825}

# Set up micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate gflow

module load eth_proxy

# Set Hugging Face cache directory
export HUGGINGFACE_HUB_CACHE=/cluster/work/cotterell/mdeprada/hf-cache

# Run training script
python train.py \
    task=openwebtext_gpt2 \
    device=gpu \
    task.training.n_samples=8 \
    task.training.accumulate_grad_batches=32 \
    task.reward.temp_end=${TEMP_END}
