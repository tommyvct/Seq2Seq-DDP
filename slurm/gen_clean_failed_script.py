
import csv
import sys



"""
The format of result.csv is like:
Name=molweni_focus_t5gemma2_270m Failed, Run time 02_00_06, TIMEOUT, ExitCode 0

model.txt is all directories in ft-models/
"""
def generate_script():
    # Read models.txt
    try:
        with open('models.txt', 'r') as f:
            all_model_dirs = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("models.txt not found")
        return

    # Read result.csv
    failed_jobs = []
    try:
        with open('result.csv', 'r') as f:
            for line in f:
                if 'Name=' in line:
                    # Extract Name=... value
                    part = line.split(' ')[0]
                    if part.startswith('Name='):
                        job_name = part.split('Name=')[1]
                        # Remove trailing comma if present (though split by space should handle it if comma follows space, but here comma likely follows name immediately? 
                        # "Name=... Failed," -> split by space gives "Name=..." and "Failed,"
                        # Let's check the file content again.
                        # "Name=molweni_natural_t5gemma2_4b Failed, Run time 00_22_45, OUT_OF_MEMORY"
                        # Split by space: ["Name=molweni_natural_t5gemma2_4b", "Failed,", ...]
                        failed_jobs.append(job_name)
    except FileNotFoundError:
        print("result.csv not found")
        return

    commands = []
    
    for job_name in failed_jobs:
        # Parse job_name
        # format: corpus_type_model_info
        # corpus: molweni / stac
        # type: natural / augmented / focus / natural2
        
        parts = job_name.split('_')
        if len(parts) < 3:
            print(f"Skipping malformed job name: {job_name}")
            continue
            
        corpus = parts[0]
        structure_type = parts[1]
        
        # reconstruct model part
        # remaining parts could be "t5gemma2", "4b" -> "t5gemma2_4b"
        # or "flan-t5", "xxl" -> "flan-t5_xxl" (Wait, split by '_' splits flan-t5? No, hyphen matches.)
        # job: stac_focus_flan-t5_base -> parts: stac, focus, flan-t5, base.
        # job: molweni_focus_t0-3b -> parts: molweni, focus, t0-3b
        
        model_parts = parts[2:]
        model_str_original = "_".join(model_parts)
        
        # Determine directory prefix for model
        dir_model_prefix = ""
        
        # Special case for t0-3b
        if "t0-3b" in model_str_original:
             # In models.txt it appears as t0-3b-3b
             dir_model_prefix = "t0-3b-3b"
        else:
            # Replace underscores with hyphens
            # e.g. t5gemma2_4b -> t5gemma2-4b
            # flan-t5_large -> flan-t5-large
            dir_model_prefix = model_str_original.replace('_', '-')
            
        # Construct the matching pattern
        # Directory format: {dir_model_prefix}_train_{corpus}_{structure_type}_
        match_pattern = f"{dir_model_prefix}_train_{corpus}_{structure_type}_"
        
        found = False
        for directory in all_model_dirs:
            if directory.startswith(match_pattern):
                commands.append(f"rm -rf {directory}")
                found = True
        
        if not found:
            print(f"Warning: No directory found for job {job_name} (Pattern: {match_pattern})")

    with open('clean_failed_jobs.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        for cmd in commands:
            f.write(cmd + "\n")
            
    print(f"Generated clean_failed_jobs.sh with {len(commands)} commands.")

if __name__ == '__main__':
    generate_script()
