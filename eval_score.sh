#!/bin/bash

# Define the list of datasets
datasets=("adult" "default" "shoppers" "magic" "beijing" "news")

# Define the list of methods
models=("arf" "ctgan" "smote" "stasy" "codi" "tabddpm" "tabsyn" "real")

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    # Iterate over each method for each dataset
    for model in "${models[@]}"; do
        echo "Evaluating MLE $model for dataset: $dataset"
        python eval/eval_mle.py --dataname "$dataset" --model "$model"

        echo "Evaluating Detection $model for dataset: $dataset"
        python eval/eval_detection.py --dataname "$dataset" --model "$model"

        echo "Evaluating Quality $model for dataset: $dataset"
        python eval/eval_quality.py --dataname "$dataset" --model "$model"

    done
done

echo "All training and sampling processes are completed."
