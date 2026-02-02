#!/bin/bash

# Loop through each directory in the ./ft-models folder
for model_dir in ./ft-models/*; do
    # Remove trailing slash
    model_dir=${model_dir%/}
    
    echo "Processing $model_dir..."

    # Find all checkpoint directories, sort them by number (descending), and skip the first one (the largest)
    # The regex extracts the number after 'checkpoint-' for sorting
    find "$model_dir" -maxdepth 1 -type d -name "checkpoint-*" | \
    sort -t- -k2 -nr | \
    tail -n +2 | \
    while read -r checkpoint_to_remove; do
        if [ -d "$checkpoint_to_remove" ]; then
            echo "  Removing $checkpoint_to_remove"
            rm -rf "$checkpoint_to_remove"
        fi
    done
done

echo "Cleanup complete."