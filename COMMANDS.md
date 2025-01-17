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




Graveyard
```bash
# Hosting on fireworks
wget -O firectl.gz https://storage.googleapis.com/fireworks-public/firectl/stable/linux-amd64.gz
gunzip firectl.gz
sudo install -o root -g root -m 0755 firectl /usr/local/bin/firectl

# Upload unsloth model
firectl 

# Deploying to fireworks
firectl set-api-key $FIREWORKS_API_KEY
firectl create model account/aidando73/models/llama-3.3-70b-instruct-code-agent-fine-tune-v1 models/llama-3.3-70b-instruct-code-agent-fine-tune-v1 --base-model 
firectl deploy aidando73/llama-3.3-70b-instruct-code-agent-fine-tune-v1
# Doesn't work I hit: 2025/01/17 08:05:23 Failed to execute: error uploading model: error reading safetensors file Llama-3.3-70B-Instruct-bnb-4bit/model-00004-of-00008.safetensors: error reading safetensors metadata: header too large: max 100000000, actual 2336927755350992246
```