import os
import replicate
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from typing import Optional

# =========================================================
# Load environment variables
# =========================================================
load_dotenv(find_dotenv())

# =========================================================
# Init A4F client
# =========================================================
A4F_API_KEY = os.getenv("api_key")
if not A4F_API_KEY:
    raise RuntimeError("A4F api_key not found")

a4f_client = OpenAI(
    base_url="https://api.a4f.co/v1",
    api_key=A4F_API_KEY,
)

# =========================================================
# Init Replicate client
# =========================================================
REPLICATE_API_TOKEN = (
    os.getenv("REPLICATE_API_TOKEN")
    or os.getenv("rep_api_key")
    or os.getenv("REP_API_KEY")
)
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN not found")

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# =========================================================
# Init Google Gemini (Prompt Enhancer)
# =========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# =========================================================
# FastAPI app + CORS
# =========================================================
app = FastAPI(title="Unified AI Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Schemas
# =========================================================
class GenerateRequest(BaseModel):
    prompt: str
    provider: str            # "replicate" | "a4f"
    aspect_ratio: Optional[str] = "16:9"
    enhance: Optional[bool] = True

class GenerateResponse(BaseModel):
    image_url: str
    used_prompt: str

# =========================================================
# Prompt Enhancer (INTERNAL)
# =========================================================
SYSTEM_PROMPT = """
You are a professional cinematic prompt engineer for AI image and video generation.

Expand short or vague prompts into rich, detailed, cinematic descriptions.
Preserve the user's intent.
Add environment, lighting, mood, camera perspective, realism, and atmosphere.
Use natural language paragraphs.
Do not include unsafe or copyrighted content.
Output only the enhanced prompt.
give only 30 words in enhancement.
"""

def enhance_prompt(user_prompt: str) -> str:
    try:
        response = gemini_model.generate_content(
            f"{SYSTEM_PROMPT}\n\nUser prompt:\n{user_prompt}"
        )
        return response.text.strip()
    except Exception:
        # Fail-safe
        return user_prompt

# =========================================================
# Unified Image Generation Endpoint
# =========================================================
@app.post("/generate-image", response_model=GenerateResponse)
def generate_image(data: GenerateRequest):

    provider = data.provider.lower()

    # 🔑 Enhance prompt internally
    final_prompt = (
        enhance_prompt(data.prompt)
        if data.enhance
        else data.prompt
    )

    # =======================
    # REPLICATE
    # =======================
    if provider == "replicate":
        try:
            output = replicate_client.run(
                "google/imagen-4",
                input={
                    "prompt": final_prompt,
                    "aspect_ratio": data.aspect_ratio,
                    "safety_filter_level": "block_medium_and_above",
                },
            )

            if isinstance(output, list):
                image_url = output[0]
            elif isinstance(output, str):
                image_url = output
            elif hasattr(output, "url"):
                image_url = output.url
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected Replicate output: {output}",
                )

            return {
                "image_url": image_url,
                "used_prompt": final_prompt,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # =======================
    # A4F
    # =======================
    elif provider == "a4f":
        try:
            size = "1280x720" if data.aspect_ratio == "16:9" else "1024x1024"

            response = a4f_client.images.generate(
                model="provider-8/z-image",
                prompt=final_prompt,
                n=1,
                response_format="url",
                size=size,
            )

            return {
                "image_url": response.data[0].url,
                "used_prompt": final_prompt,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid provider. Use 'replicate' or 'a4f'",
        )
