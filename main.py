import os
import random
import logging
from typing import Literal, Optional, List

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, OpenAIError

# =========================================================
# Logging Configuration - Comprehensive error tracking
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================================================
# Load Environment Variables
# =========================================================
try:
    load_dotenv(find_dotenv())
    logger.info(" Environment variables loaded successfully")
except Exception as e:
    logger.error(f" Failed to load .env file: {e}")
    raise

# =========================================================
# A4F Configuration - Single model only: provider-4/imagen-3.5
# =========================================================
SINGLE_MODEL_ID = "provider-8/z-image"  # Fixed model as requested
A4F_BASE_URL = os.getenv("A4F_BASE_URL", "https://api.a4f.co/v1").rstrip("/")

try:
    raw_keys = os.getenv("A4F_API_KEYS") or os.getenv("api_key") or ""
    A4F_API_KEYS: List[str] = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not A4F_API_KEYS:
        raise RuntimeError("No A4F API keys found. Set A4F_API_KEYS or api_key in .env")
    
    logger.info(f" Loaded {len(A4F_API_KEYS)} A4F API key(s)")
    logger.info(f" Using fixed model: {SINGLE_MODEL_ID}")
    logger.info(f" A4F Base URL: {A4F_BASE_URL}")
except Exception as e:
    logger.critical(f" API configuration failed: {e}")
    raise

# Initialize single OpenAI client per API key for load balancing
try:
    a4f_clients: List[OpenAI] = [
        OpenAI(base_url=A4F_BASE_URL, api_key=key) for key in A4F_API_KEYS
    ]
    logger.info(f" Initialized {len(a4f_clients)} A4F client(s)")
except Exception as e:
    logger.critical(f" Failed to initialize A4F clients: {e}")
    raise

def pick_a4f_client() -> OpenAI:
    """Randomly select A4F client for load distribution across API keys."""
    return random.choice(a4f_clients)

# =========================================================
# Google Gemini Prompt Enhancer Initialization
# =========================================================
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY missing from .env")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    logger.info("Google Gemini prompt enhancer initialized")
except Exception as e:
    logger.critical(f" Google Gemini initialization failed: {e}")
    raise

# =========================================================
# FastAPI Application Setup
# =========================================================
app = FastAPI(title="Unified AI Image Generator - Imagen 3.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(" FastAPI app + CORS configured")

# =========================================================
# Pydantic Schemas
# =========================================================
StyleId = Literal["none", "cinematic", "photorealistic", "anime", "fantasy"]

class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: str  # "16:9" | "9:16" | "1:1" | "4:3" | "21:9"
    style_preset: StyleId
    enhance: bool = False

class GenerateResponse(BaseModel):
    image_url: str
    used_prompt: str
    used_provider: str = "provider-4"
    used_model: str = SINGLE_MODEL_ID

class EnhanceRequest(BaseModel):
    prompt: str
    style_preset: StyleId

class EnhanceResponse(BaseModel):
    enhanced_prompt: str

# =========================================================
# Style-Specific Enhancement Instructions (Base + Style)
# =========================================================
def build_style_system_prompt(style: StyleId) -> str:
    """Builds base professional prompt + style-specific instructions."""
    base_professional = (
        "Generate a clear, high-quality, professional image that closely follows the user's intent. "
        "Include detailed environment, natural lighting, proper composition, and high-resolution details."
    )
    
    style_instructions = {
        "cinematic": "Cinematic style: dramatic lighting, shallow depth of field, film-like composition, expressive atmosphere, 4K detail.",
        "photorealistic": "Photorealistic style: ultra-realistic camera optics, natural lighting, accurate materials/textures, photographic detail.",
        "anime": "Anime style: clean line art, vibrant colors, expressive characters, studio-quality shading and highlights.",
        "fantasy": "Fantasy style: epic worldbuilding, magical elements, dramatic lighting, highly detailed 4K rendering.",
        "none": ""
    }
    
    style_part = style_instructions.get(style, "")
    return f"{base_professional} {style_part}".strip()

# =========================================================
# Prompt Enhancement with Gemini (Style-Aware)
# =========================================================
BASE_SYSTEM_PROMPT = """
You are an expert AI image prompt engineer.

CRITICAL RULES:
- Transform short/vague prompts into vivid, 30-50 word descriptions
- Preserve 100% of user's original intent - do NOT change the subject
- Add: environment, lighting, mood, camera angle, composition details
- Natural flowing sentences only (NO bullets, lists, or keywords)
- Professional quality: high detail, proper proportions, realistic physics
- Output ONLY the enhanced prompt text. Nothing else.
""".strip()

def enhance_prompt(user_prompt: str, style: StyleId) -> str:
    """Enhance user prompt using Gemini with professional base + style instructions."""
    try:
        logger.info(f" Enhancing: '{user_prompt[:40]}...' (style: {style})")
        
        style_prompt = build_style_system_prompt(style)
        full_prompt = f"{BASE_SYSTEM_PROMPT}\n\nStyle guidance:\n{style_prompt}\n\nUser prompt:\n{user_prompt}"
        
        response = gemini_model.generate_content(full_prompt)
        enhanced = response.text.strip()
        
        if enhanced:
            logger.info(f"Enhanced: '{enhanced[:40]}...'")
            return enhanced
        else:
            logger.warning(" Gemini empty response, fallback to original")
            return user_prompt
            
    except Exception as e:
        logger.error(f" Gemini enhancement failed: {e}")
        return user_prompt  # Graceful fallback

# =========================================================
# Aspect Ratio to A4F Size Mapping
# =========================================================
def aspect_ratio_to_size(ratio: str) -> str:
    """Convert aspect ratio to A4F-compatible size string."""
    mapping = {
        "16:9": "1280x720",
        "9:16": "1024x1792", 
        "1:1": "1024x1024",
        "4:3": "1024x768",
        "21:9": "1920x820"
    }
    return mapping.get(ratio, "1024x1024")

# =========================================================
# Main Image Generation Endpoint
# =========================================================
@app.post("/generate-image", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """
    Workflow:
    1. Enhance prompt with Gemini (if requested) using base professional + style prompt
    2. Call A4F provider-4/imagen-3.5 with random API key
    3. Detailed error logging for API/Model/Rate-limit failures
    """
    logger.info(f"  Generation: '{request.prompt[:40]}...' | Style: {request.style_preset} | Ratio: {request.aspect_ratio} | Enhance: {request.enhance}")
    
    try:
        # 1. Prompt preparation (enhance or use original)
        final_prompt = enhance_prompt(request.prompt, request.style_preset) if request.enhance else request.prompt
        size = aspect_ratio_to_size(request.aspect_ratio)
        client = pick_a4f_client()
        
        logger.info(f" Calling A4F: model={SINGLE_MODEL_ID}, size={size}, client={hash(client)}")
        
        # 2. A4F Image Generation API Call
        response = client.images.generate(
            model=SINGLE_MODEL_ID,
            prompt=final_prompt,
            n=1,
            response_format="url",
            size=size,
        )
        
        image_url = response.data[0].url
        logger.info(f" SUCCESS: Image generated - {image_url[:60]}...")
        
        return GenerateResponse(
            image_url=image_url,
            used_prompt=final_prompt,
            used_provider="provider-4/imagen-3.5",
            used_model=SINGLE_MODEL_ID
        )
        
    # 3. Precise API Error Handling with Detailed Logging
    except OpenAIError as api_error:
        error_message = str(api_error)
        logger.error(f" A4F API ERROR: {error_message}")
        logger.error(f"   Model: {SINGLE_MODEL_ID} | Prompt len: {len(final_prompt)} | Size: {size}")
        
        if api_error.status_code == 400:
            logger.error("    Likely: Invalid model/prompt/size or unsupported params")
            raise HTTPException(400, f"Invalid request (Model/Prompt error): {error_message}")
        elif api_error.status_code == 401:
            logger.error("    API Key authentication failure")
            raise HTTPException(401, "Invalid API key - check A4F_API_KEYS in .env")
        elif api_error.status_code == 429:
            logger.error("    Rate limit exceeded")
            raise HTTPException(429, "Rate limit hit. Try again in 1-2 minutes.")
        elif api_error.status_code == 404:
            logger.error("    Model not found: provider-4/imagen-3.5 unavailable")
            raise HTTPException(404, f"Model {SINGLE_MODEL_ID} not available on A4F")
        else:
            logger.error(f"    Unexpected API status: {api_error.status_code}")
            raise HTTPException(500, f"A4F API failed: {error_message}")
    
    except Exception as e:
        logger.critical(f" UNEXPECTED ERROR in generate_image: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error during image generation")

# =========================================================
# Prompt Enhancement Endpoint (Standalone)
# =========================================================
@app.post("/enhance-prompt", response_model=EnhanceResponse)
async def enhance_prompt_endpoint(request: EnhanceRequest):
    """Standalone prompt enhancement using professional base + style."""
    try:
        enhanced = enhance_prompt(request.prompt, request.style_preset)
        logger.info(f" Standalone enhancement complete")
        return EnhanceResponse(enhanced_prompt=enhanced)
    except Exception as e:
        logger.error(f" Enhance endpoint failed: {e}", exc_info=True)
        raise HTTPException(500, "Prompt enhancement failed")

# =========================================================
# Health Check with Detailed Diagnostics
# =========================================================
@app.get("/health")
async def health_check():
    """Health check with configuration diagnostics."""
    return {
        "status": "healthy",
        "model": SINGLE_MODEL_ID,
        "api_keys_count": len(A4F_API_KEYS),
        "base_url": A4F_BASE_URL,
        "gemini_ready": True
    }

# =========================================================
# Application Lifecycle Events
# =========================================================
@app.on_event("startup")
async def startup():
    logger.info("=" * 70)
    logger.info(" A4F Imagen 3.5 Image Generator STARTED")
    logger.info(f"   Model: {SINGLE_MODEL_ID}")
    logger.info(f"   Keys: {len(A4F_API_KEYS)}")
    logger.info(f"   Base: {A4F_BASE_URL}")
    logger.info("=" * 70)

@app.on_event("shutdown")
async def shutdown():
    logger.info("=" * 70)
    logger.info(" Application SHUTTING DOWN")
    logger.info("=" * 70)
