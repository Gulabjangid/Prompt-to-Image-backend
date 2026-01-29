Backend FastAPI service

- Copy `backend/.env.example` to `backend/.env` and set `MINIMAX_API_KEY`.

Run locally (from workspace root):

```bash
pip install -r requirement.txt
uvicorn backend.main:app --reload --port 8000
```

Endpoints:

- POST `/v1/video_generation` - create a video generation task (expects JSON body per OpenAPI).
- POST `/generate-image` - example image-generation endpoint (existing).

Notes:
- The video generation handler reads `MINIMAX_API_KEY` at request time.
- Adjust CORS and security for production.
