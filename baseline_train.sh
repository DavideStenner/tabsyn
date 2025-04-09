#!/bin/bash

# Define the list of datasets
datasets=("adult" "default" "shoppers" "magic" "beijing" "news")

# Define the list of methods
methods=("arf" "smote" "goggle" "great" "stasy" "codi" "tabddpm")

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    # Iterate over each method for each dataset
    for method in "${methods[@]}"; do
        echo "Training $method for dataset: $dataset"
        python main.py --dataname "$dataset" --method "$method" --mode train
        
        echo "Sampling with $method for dataset: $dataset"
        save_path="synthetic/$dataset/$method.csv"
        python main.py --dataname "$dataset" --method "$method" --mode sample --save_path "$save_path"
    done
done

echo "All training and sampling processes are completed."
