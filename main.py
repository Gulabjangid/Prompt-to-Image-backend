import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.a4f.co/v1",
    api_key=os.getenv("api_key"),
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (important for Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ImageRequest(BaseModel):
    prompt: str

# Response schema
class ImageResponse(BaseModel):
    image_url: str

@app.post("/generate-image", response_model=ImageResponse)
def generate_image(data: ImageRequest):
    try:
        response = client.images.generate(
            model="provider-4/imagen-3.5",
            prompt=data.prompt,
            n=1,
            response_format="url",
            size="1024x1024"
        )

        image_url = response.data[0].url
        return {"image_url": image_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
