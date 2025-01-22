# Mock a vLLM server

```bash
# Setup
source ~/miniconda3/bin/activate ./mock
conda create --prefix ./mock python=3.10
source ~/miniconda3/bin/activate ./mock
pip install fastapi pydantic uvicorn
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Run
python mock.py
```