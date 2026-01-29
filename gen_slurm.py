import shlex
import os

def get_resources(tokens):
    t_val = None
    m_val = None
    
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
            
    # Group 1: Specific small models
    # (flan-t5-base, flan-t5-large, t5-large, t5gemma2-270m)
    g1_pairs = [
        ('flan-t5', 'base'),
        ('flan-t5', 'large'),
        ('t5', 'large'),
        ('t5gemma2', '270m')
    ]
    
    if (t_val, m_val) in g1_pairs:
        return "--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=4 --mem=8G --time=02:00:00"
        
    # Group 2: Large models (11b or xxl)
    if m_val in ['11b', 'xxl']:
         return "--gpus=h100:2 --cpus-per-task=4 --mem=32G --time=24:00:00"
         
    # Group 3: All others
    return "--gpus=h100:1 --cpus-per-task=4 --mem=16G --time=04:00:00"

def get_job_name(tokens):
    parts = []
    # Extract corpus, setup, type, size for name
    if '--train_corpus' in tokens: parts.append(tokens[tokens.index('--train_corpus')+1])
    if '-s' in tokens: parts.append(tokens[tokens.index('-s')+1])
    if '-t' in tokens: parts.append(tokens[tokens.index('-t')+1])
    if '-m' in tokens: parts.append(tokens[tokens.index('-m')+1])
    
    if parts:
        return "_".join(parts)
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
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH {resources}

source init_hpc.sh
{line}
EOT
"""
            print(script_block)

if __name__ == "__main__":
    main()
