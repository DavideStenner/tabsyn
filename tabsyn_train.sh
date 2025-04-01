#!/bin/bash

# Define the list of datasets
datasets=("adult" "default" "shoppers" "magic" "beijing" "news")

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    echo "Training VAE for dataset: $dataset"
    python main.py --dataname "$dataset" --method vae --mode train

    echo "Training diffusion model (TabSyn) for dataset: $dataset"
    python main.py --dataname "$dataset" --method tabsyn --mode train

    echo "Sampling with diffusion model (TabSyn) for dataset: $dataset"
    save_path="synthetic/$dataset/tabsyn.csv"
    python main.py --dataname "$dataset" --method tabsyn --mode sample --save_path "$save_path"
done

echo "All training and sampling processes are completed."
