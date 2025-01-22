```bash
# Mock a vLLM server
conda create --prefix ./mock python=3.10
source ~/miniconda3/bin/activate ./mock
pip install fastapi pydantic uvicorn
```