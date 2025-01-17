from vllm import LLM, SamplingParams
import torch

llm = LLM(
    model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    max_model_len=100_000,
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

prompts = [
    "Hello World"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output)
