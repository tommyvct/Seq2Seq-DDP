#!/bin/bash

BASE_DIR="ft-models"
DISCARD_DIR="discard"

# Create discard directory if it doesn't exist
mkdir -p "$DISCARD_DIR"

for model_dir in "$BASE_DIR"/*/; do
    # Remove trailing slash
    model_dir=${model_dir%/}
    model_name=$(basename "$model_dir")
    
    echo "Processing $model_name..."

    # Gather all checkpoint folders and sort them numerically
    # The folders are named checkpoint-<number>
    checkpoints=($(ls -d "$model_dir"/checkpoint-*/ 2>/dev/null | sort -V))

    if [ ${#checkpoints[@]} -gt 1 ]; then
        # The last element is the highest numbered checkpoint
        latest_checkpoint="${checkpoints[-1]}"
        echo "  Keeping: $latest_checkpoint"

        # Create model-specific discard folder
        model_discard_dir="$DISCARD_DIR/$model_name"
        mkdir -p "$model_discard_dir"

        # Loop through all but the last checkpoint
        unset 'checkpoints[${#checkpoints[@]}-1]'
        for cp in "${checkpoints[@]}"; do
            echo "  Moving $cp to $model_discard_dir/"
            mv "$cp" "$model_discard_dir/"
        done
    else
        echo "  Found only ${#checkpoints[@]} checkpoint(s). Skipping."
    fi
done
