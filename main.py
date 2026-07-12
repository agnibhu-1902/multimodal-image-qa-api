import base64
import os

from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageQARequest(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
async def answer_image(request: ImageQARequest):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        extra_body={
            "models": [
                "openai/gpt-oss-20b:free",
                "google/gemma-4-31b-it:free",
                "google/gemma-4-26b-a4b-it:free",
                "meta-llama/llama-4-maverick:free",
            ],
            "route": "fallback",
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{request.image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Look at this image carefully and answer the question below.\n"
                            f"Return ONLY the answer value — no extra words, no units, no currency symbols.\n"
                            f"If the answer is numeric, return just the number like: 4089.35\n"
                            f"Do not add any explanation.\n\n"
                            f"Question: {request.question}"
                        ),
                    },
                ],
            }
        ],
    )

    answer = response.choices[0].message.content.strip()
    return {"answer": answer}
