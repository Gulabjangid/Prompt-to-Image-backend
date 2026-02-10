"""
  pip i bytez
"""

from bytez import Bytez

key = "91ca4f9085cf3484b8513e9c6b87fa58"
sdk = Bytez(key)

# choose stable-diffusion-xl-base-1.0
model = sdk.model("google/imagen-4.0-ultra-generate-001")

# send input to model
results = model.run("A pretty girl hot in her 30s with long hair and blue eyes, photorealistic, 4k, high quality")

print({ "error": results.error, "output": results.output })