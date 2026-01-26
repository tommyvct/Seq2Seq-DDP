export CUDA_HOME=/opt/pytorch/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

python3 train.py --do_train -s natural -t t0-3b -m 270m -b