#!/bin/bash

# Loop through each directory in the ./ft-models folder
for model_dir in ./ft-models/*; do
    # Remove trailing slash
    model_dir=${model_dir%/}
    
    echo "Processing $model_dir..."

    # Enter directory to safely handle paths with hyphens
    pushd "$model_dir" > /dev/null

    # Find all checkpoint directories in current folder
    # Use version sort (-V) to handle numbers correctly (e.g. 10 > 9), reverse (-r) so largest is first
    find . -maxdepth 1 -type d -name "checkpoint-*" | \
    sort -Vr | \
    tail -n +2 | \
    while read -r checkpoint_to_remove; do
        if [ -d "$checkpoint_to_remove" ]; then
            echo "  Removing $checkpoint_to_remove"
            rm -rf "$checkpoint_to_remove"
        fi
    done

    # Return to previous directory
    popd > /dev/null
done

echo "Cleanup complete."