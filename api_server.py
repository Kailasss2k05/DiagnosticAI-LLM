import os
from threading import Lock

from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv("MODEL_PATH", "model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))


app = FastAPI(title="Fine-Tuned Model Test API", version="1.0.0")

_llm = None
_llm_lock = Lock()


class GenerateRequest(BaseModel):
    instruction: str = Field(default="Answer the electrician query")
    input: str = Field(..., min_length=1, description="User issue description")
    max_tokens: int = Field(default=120, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    prompt: str
    response: str


def build_prompt(instruction: str, user_input: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Input:\n"
        f"{user_input.strip()}\n\n"
        "### Response:\n"
    )


def get_llm() -> Llama:
    global _llm
    if _llm is not None:
        return _llm

    with _llm_lock:
        if _llm is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
            _llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX)
    return _llm


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/test", response_model=GenerateResponse)
def test_model(payload: GenerateRequest) -> GenerateResponse:
    prompt = build_prompt(payload.instruction, payload.input)

    try:
        llm = get_llm()
        output = llm(
            prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            stop=["\n---", "\n### Instruction:"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    raw_text = output["choices"][0]["text"].strip()

    return GenerateResponse(
        prompt=prompt,
        response=raw_text,
    )