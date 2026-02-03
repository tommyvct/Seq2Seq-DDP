sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_molweni_natural2
#SBATCH --output=t0-3b_molweni_natural2_%j.out
#SBATCH --error=t0-3b_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT


sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_molweni_natural2
#SBATCH --output=flan-t5_xl_molweni_natural2_%j.out
#SBATCH --error=flan-t5_xl_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_molweni_natural2
#SBATCH --output=t5_3b_molweni_natural2_%j.out
#SBATCH --error=t5_3b_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t0-3b_molweni_focus
#SBATCH --output=t0-3b_molweni_focus_%j.out
#SBATCH --error=t0-3b_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t0-3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_xl_molweni_focus
#SBATCH --output=flan-t5_xl_molweni_focus_%j.out
#SBATCH --error=flan-t5_xl_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m xl -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_3b_molweni_focus
#SBATCH --output=t5_3b_molweni_focus_%j.out
#SBATCH --error=t5_3b_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=16G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m 3b -l 5e-5 -e 3 --batchsize 4 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_molweni_focus
#SBATCH --output=flan-t5_large_molweni_focus_%j.out
#SBATCH --error=flan-t5_large_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=8G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_molweni_natural2
#SBATCH --output=t5_large_molweni_natural2_%j.out
#SBATCH --error=t5_large_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=8G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=flan-t5_large_molweni_natural2
#SBATCH --output=flan-t5_large_molweni_natural2_%j.out
#SBATCH --error=flan-t5_large_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=8G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t flan-t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_molweni_natural2
#SBATCH --output=t5gemma2_270m_molweni_natural2_%j.out
#SBATCH --error=t5gemma2_270m_molweni_natural2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=8G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s natural2 -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5gemma2_270m_molweni_focus
#SBATCH --output=t5gemma2_270m_molweni_focus_%j.out
#SBATCH --error=t5gemma2_270m_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=8G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5gemma2 -m 270m -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=t5_large_molweni_focus
#SBATCH --output=t5_large_molweni_focus_%j.out
#SBATCH --error=t5_large_molweni_focus_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wut2@unbc.ca
#SBATCH --gpus=h100:1 --cpus-per-task=4 --mem=8G --time=08:00:00

source init_hpc.sh
python3 train.py --train_corpus molweni --do_train -s focus -t t5 -m large -l 5e-5 -e 3 --batchsize 8 --step 2000 -b
EOT