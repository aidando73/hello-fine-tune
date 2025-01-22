
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

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
                "id": "mock-model",
                "object": "model",
                "created": 1677610602,
                "owned_by": "mock"
            }
        ]
    }

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    return CompletionResponse(
        id="mock-completion",
        object="text_completion",
        created=1677610602,
        model=request.model,
        choices=[{
            "text": "This is a mock response",
            "index": 0,
            "logprobs": None,
            "finish_reason": "length"
        }]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
