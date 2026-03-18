module load arrow
module load python/3.12.4
module load cuda

export HF_HOME=$SCRATCH/huggingface


if [ ! -d "venv" ]; then
    virtualenv --no-download venv
    source venv/bin/activate
    pip install jellyfish
    pip install torch torchvision torchaudio --no-index
    pip install -r requirements.txt --no-index
    ./run_dataprocess.sh
    python3 download_nltk_punkt.py
else
    source venv/bin/activate
fi
