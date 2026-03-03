import shlex
import os

def get_resources(tokens):
    t_val = None
    m_val = None
    molweni = False
    transition = False

    CONFIG_molweni_time_multiplier = 2
    
    if '-t' in tokens:
        try:
            t_val = tokens[tokens.index('-t') + 1]
        except IndexError:
            pass
            
    if '-m' in tokens:
        try:
            m_val = tokens[tokens.index('-m') + 1]
        except IndexError:
            pass

    if '--train_corpus' in tokens:
        try:
            molweni = tokens[tokens.index('--train_corpus') + 1] == 'molweni'
        except IndexError:
            pass
    if '-s' in tokens:
        try:
            transition = tokens[tokens.index('-s') + 1] in ['focus', 'natural2']
        except IndexError:
            pass
            
    # Group 1: Specific small models
    # (flan-t5-base, flan-t5-large, t5-large, t5gemma2-270m)
    g1_pairs = [
        # ('flan-t5', 'base'),
        # ('flan-t5', 'large'),
        # ('t5', 'large'),
        ('t5gemma2', '270m')
    ]
    
    if (t_val, m_val) in g1_pairs:
        time_in_hours = 6
            
        return f"--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=16G --time={time_in_hours:02d}:00:00"
        
    # Group 2: Medium models
    # (t5gemma2-1b, t5-3b, flan-t5-xl)
    g2_pairs = [
        ('t5gemma2', '1b'),
        # ('t5', '3b'),
        # ('flan-t5', 'xl')
    ]

    if (t_val, m_val) in g2_pairs:
        time_in_hours = 12
        if not transition:
            time_in_hours = 6
        if molweni:
            time_in_hours *= CONFIG_molweni_time_multiplier
            
        return f"--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=8 --mem=32G --time={time_in_hours:02d}:00:00"

    # Group 3: Large models
    # (t5gemma2-4b)
    g3_pairs = [
        ('t5gemma2', '4b')
    ]

    if (t_val, m_val) in g3_pairs:
        time_in_hours = 12
        if not transition:
            time_in_hours = 6
        if molweni:
            time_in_hours *= CONFIG_molweni_time_multiplier
            
        return f"--gpus=h100:1 --cpus-per-task=16 --mem=64G --time={time_in_hours:02d}:00:00"

    # Group 4: Extra Large models (11b or xxl)
    # if m_val in ['11b', 'xxl']:
    #     return f"--gpus=h100:1 --cpus-per-task=8 --mem=32G --time={time_in_hours:02d}:00:00"
         
    # Default fallback (t0-3b)
    time_in_hours = 12
    if not transition:
        time_in_hours = 6
    if molweni:
        time_in_hours *= CONFIG_molweni_time_multiplier
    return f"--gpus=h100:1 --cpus-per-task=8 --mem=64G --time={time_in_hours:02d}:00:00"

def get_job_name(tokens):
    parts = []
    # Extract corpus, setup, type, size for name
    if '-t' in tokens: parts.append(tokens[tokens.index('-t')+1])
    if '-m' in tokens: parts.append(tokens[tokens.index('-m')+1])
    if '--train_corpus' in tokens: parts.append(tokens[tokens.index('--train_corpus')+1])
    if '-s' in tokens: parts.append(tokens[tokens.index('-s')+1])
    
    if parts:
        return "train_" + "_".join(parts)
    return "train_job"

def main():
    with open('run_train.sh', 'r') as f:
        lines = f.readlines()

    print("#!/bin/bash")
    print("# Auto-generated submission script")
    print("")

    for line in lines:
        line = line.strip()
        if line.startswith('python3'):
            tokens = shlex.split(line)
            resources = get_resources(tokens)
            job_name = get_job_name(tokens)
            
            # Escape $ characters in the python command if any (unlikely in this context but good practice)
            # But the 'source init_hpc.sh' needs to be literal.
            # Using quoted heredoc 'EOT' prevents expansion in the block, so we don't need to escape $ inside.
            # But we want sbatch to evaluate the directives. sbatch reads #SBATCH lines.
            
            script_block = f"""sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm/logs/train/{job_name}_%j.out
#SBATCH --error=slurm/logs/train/{job_name}_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH {resources}

cd $HOME/scratch/Seq2Seq-DDP
source slurm/init_hpc.sh
{line}
EOT
"""
            print(script_block)

if __name__ == "__main__":
    main()
