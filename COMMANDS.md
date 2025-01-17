```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10

source ~/miniconda3/bin/activate ./env
pip install -r requirements.txt

# Removing CUDA 12.4 (default on lambda labs)
sudo apt-get remove --auto-remove nvidia-cuda-toolkit
sudo apt-get purge nvidia-cuda-toolkit
# ^ Wait don't think it's required - https://deeptalk.lambdalabs.com/t/downgrading-cuda-12-4-to-12-1/4385

conda create --prefix ./env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate ./env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```
