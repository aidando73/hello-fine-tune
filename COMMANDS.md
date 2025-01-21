```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10

tmux new-session -d -s fine-tune
tmux switch-client -t fine-tune
conda create --prefix ./env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
source ~/miniconda3/bin/activate ./env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt

pip install --no-deps trl peft accelerate bitsandbytes
git clone https://huggingface.co/aidando73/llama-3.3-70b-instruct-code-agent-fine-tune-v1

# Serving with vllm
conda create --prefix ./vllm python=3.12 -y
conda activate ./vllm
pip install vllm
pip install bitsandbytes>=0.45.0
# base server
vllm serve unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --port 8000 \
    --quantization bitsandbytes \
    --dtype bfloat16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-model-len 50_000 \
    --load-format bitsandbytes
# with Lora adapter
vllm serve unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --enable-lora \
    --lora-modules v1=$(pwd)/models/llama-3.3-70b-instruct-code-agent-fine-tune-v1 \
    --port 8000 \
    --quantization bitsandbytes \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 50_000 \
    --load-format bitsandbytes

# Serving with vllm
screen -S agent-eval
source ~/miniconda3/bin/activate
conda create --prefix ./vllm python=3.12 -y
conda activate ./vllm
pip install vllm bitsandbytes>=0.45.0 && apt-get update && apt-get install -y build-essential

source ~/miniconda3/bin/activate ./vllm
# lora adapter from huggingface
vllm serve unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --enable-lora \
    --lora-modules v1=aidando73/llama-3.3-70b-instruct-code-agent-fine-tune-v1 \
    --port 8000 \
    --quantization bitsandbytes \
    --dtype bfloat16 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --max-model-len 50_000 \
    --load-format bitsandbytes

curl http://localhost:8000/v1/models | jq .

# Converting from HuggingFace model to GGUF format
conda create --prefix ./llama-cpp python=3.12 -y
source ~/miniconda3/bin/activate ./llama-cpp
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt

python llama.cpp/convert_hf_to_gguf.py \
  --outfile llama-3.3-70b-instruct-code-agent-fine-tune-v1-gguf-q8_0 \
  --outtype q8_0 \
  --split-max-size 30G \
  aidando73/llama-3.3-70b-instruct-code-agent-fine-tune-v1-merged

python llama.cpp/convert_hf_to_gguf.py -h

conda create --prefix ./vllm python=3.12 -y
source ~/miniconda3/bin/activate ./vllm
pip install vllm
vllm serve $(realpath llama-3.3-70b-instruct-code-agent-fine-tune-v1-base-4b-quantized) \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-model-len 50_000

# Serving from huggingface
vllm serve aidando73/llama-3.3-70b-instruct-code-agent-fine-tune-v1-base-4b-quantized \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 50_000

# Serve merged LORA adapter
vllm serve aidando73/llama-3.3-70b-instruct-code-agent-fine-tune-v1-merged \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 50_000

curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
  sudo gpg --dearmor -o /etc/apt/keyrings/ngrok.gpg && \
  echo "deb [signed-by=/etc/apt/keyrings/ngrok.gpg] https://ngrok-agent.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list && \
  sudo apt update && sudo apt install ngrok

ngrok config add-authtoken __token__

ngrok http http://localhost:8000

curl https://0168-149-7-4-156.ngrok-free.app/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/dev/hello-fine-tune/llama-3.3-70b-instruct-code-agent-fine-tune-v1-base-4b-quantized",
    "prompt": "Hello, how are you?",
    "temperature": 0.7,
    "max_tokens": 500
  }'

curl https://0168-149-7-4-156.ngrok-free.app/v1/models


# Default is 4 A100s
# 500 error
firectl create deployment \
  accounts/aidando73-e35261/models/llama-3-3-70b-instruct-code-agent-fine-tune-v1-base-4b-quant

# Can try this too:
# 500s
firectl create deployment \
  --accelerator-count 4 \
  --accelerator-type NVIDIA_H100_80GB \
  accounts/aidando73-e35261/models/llama-3-3-70b-instruct-code-agent-fine-tune-v1-base-4b-quant

# 500s
firectl create deployment \
  --accelerator-count 8 \
  --accelerator-type NVIDIA_H100_80GB \
  accounts/aidando73-e35261/models/llama-3-3-70b-instruct-code-agent-fine-tune-v1-base-4b-quant

ngrok tunnel --label edge=edghts_2rvQ8TppmhPHbOkTFwqMWH2igyX http://localhost:8000
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
# ^ Need to download from lfs

```