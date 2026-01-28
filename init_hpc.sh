module load python/3.12.4
module load arrow
module load cuda

export HF_HOME=$HOME/scratch


if [ ! -d "venv" ]; then
    virtualenv --no-download venv
fi
source venv/bin/activate
pip install jellyfish
pip install torch torchvision torchaudio --no-index
pip install -r requirements.txt --no-index
