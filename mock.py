
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

from unsloth import FastLanguageModel

MODEL_ID = "aidando73/llama-3.3-70b-instruct-code-agent-fine-tune-v1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 2048,
    dtype = "float16",
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]


@app.get("/v1/models")
async def get_models():
    return {
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1677610602,
                "owned_by": "mock"
            }
        ]
    }

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    prompt = request.prompt

    input_ids = tokenizer.encode(prompt, return_tensors = "pt").to("cuda")
    output = model.generate(input_ids, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)
    res = tokenizer.decode(output[0], skip_special_tokens = True)

    return CompletionResponse(
        id="mock-completion",
        object="text_completion",
        created=1677610602,
        model=request.model,
        choices=[{
            "text": res,
            "index": 0,
            "logprobs": None,
            "finish_reason": "length"
        }]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
