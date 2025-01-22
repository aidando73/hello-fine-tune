# Mock a vLLM server

```bash
# Setup
source ~/miniconda3/bin/activate
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
source ~/miniconda3/bin/activate unsloth_env
pip install fastapi pydantic uvicorn
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Run
python mock.py
# uvicorn mock:app --host 0.0.0.0 --workers 1 --limit-concurrency 1 --timeout-keep-alive 0
```