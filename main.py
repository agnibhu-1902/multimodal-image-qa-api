import base64
import os
from io import BytesIO

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

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
    image_bytes = base64.b64decode(request.image_base64)
    image = Image.open(BytesIO(image_bytes))

    prompt = f"""Look at this image carefully and answer the question below.
Return ONLY the answer value — no extra words, no units, no currency symbols.
If the answer is numeric (e.g. a total, a score, a price), return just the number like: 4089.35
Do not add any explanation.

Question: {request.question}"""

    response = model.generate_content([prompt, image])
    return {"answer": response.text.strip()}
