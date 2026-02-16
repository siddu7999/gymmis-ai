from dotenv import load_dotenv
load_dotenv()

# estimator_service.py
# FastAPI server for Gymmis AI estimator (prod-ready tweaks).

import os
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from .ai_model import FoodEstimator

# ========= Env =========
AI_SHARED_TOKEN = os.getenv("AI_SHARED_TOKEN")                   # MUST match Ubuntu proxy
PORT          = int(os.getenv("PORT", "8100"))
CORS_ORIGINS  = os.getenv("CORS_ORIGINS", "*")
AI_PRELOAD    = os.getenv("AI_PRELOAD", "0") in {"1", "true", "True"}
AI_MAX_MB     = float(os.getenv("AI_MAX_MB", "8"))               # reject very large uploads
ALLOW_WEBP    = os.getenv("ALLOW_WEBP", "1") in {"1", "true", "True"}

# ========= App =========
app = FastAPI(title="Gymmis AI Estimator", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Lazy model =========
_estimator: Optional[FoodEstimator] = None
_model_names: Optional[list[str]] = None

def _get_estimator() -> FoodEstimator:
    global _estimator, _model_names
    if _estimator is None:
        est = FoodEstimator()
        _estimator = est
        # Optional: expose which sub-models the estimator loaded (if you printed them there)
        # Fall back to empty list if not available.
        try:
            _model_names = getattr(est, "model_names", None) or []
        except Exception:
            _model_names = []
    return _estimator

# ========= Startup (optional preload) =========
@app.on_event("startup")
async def _maybe_preload() -> None:
    if AI_PRELOAD:
        try:
            _ = _get_estimator()
            print("[estimator] model preloaded at startup")
        except Exception as e:
            # Keep the service up; /health will show model_loaded=false.
            print(f"[estimator] preload failed: {e}")

# ========= Helpers =========
def _too_big(num_bytes: int) -> bool:
    return (num_bytes / (1024 * 1024.0)) > AI_MAX_MB

def _allowed_content_type(ct: str | None) -> bool:
    if not ct:
        return False
    ct = ct.lower()
    allowed = {"image/jpeg", "image/jpg", "image/png"}
    if ALLOW_WEBP:
        allowed.add("image/webp")
    return ct in allowed

# ========= Endpoints =========
@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Liveness endpoint. Returns quickly even if the model hasn't loaded yet.
    """
    return {
        "ok": True,
        "model_loaded": _estimator is not None,
        "preload": AI_PRELOAD,
        "max_mb": AI_MAX_MB,
        "allow_webp": ALLOW_WEBP,
        "models": _model_names or [],
        "version": app.version,
    }

@app.get("/ready")
async def ready() -> Dict[str, Any]:
    """
    Readiness endpoint. Consider 'ready' only when the estimator is built.
    """
    return {"ready": _estimator is not None}

@app.post("/estimate")
async def estimate(
    request: Request,
    image: UploadFile = File(...),
    x_api_token: Optional[str] = Header(default=None),  # FastAPI maps 'x-api-token' -> x_api_token
):
    # ---- auth ----
    if not AI_SHARED_TOKEN:
        raise HTTPException(status_code=500, detail="AI_SHARED_TOKEN not set")
    if x_api_token != AI_SHARED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    # ---- content type ----
    if not _allowed_content_type(image.content_type):
        allowed = "jpeg, jpg, png" + (", webp" if ALLOW_WEBP else "")
        raise HTTPException(status_code=415, detail=f"Only {allowed} supported")

    # ---- size guard (when available) ----
    # Note: Some servers/clients don’t send Content-Length; we still check after read.
    try:
        raw = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    if _too_big(len(raw)):
        raise HTTPException(
            status_code=413,
            detail=f"Payload too large: {(len(raw)/1048576):.2f} MB > {AI_MAX_MB:.2f} MB",
        )

    # ---- run inference ----
    try:
        t0 = time.perf_counter()
        est = _get_estimator()
        result = est.estimate(image_bytes=raw)  # -> dict: dish_name, items, preview_image_id
        t_ms = int((time.perf_counter() - t0) * 1000)

        # Backward compatible: keep the main fields unchanged; add meta for observability.
        result = dict(result)  # ensure it's a plain dict
        result["meta"] = {
            "latency_ms": t_ms,
            "model_loaded": True,
            "client_ip": request.client.host if request.client else None,
            "content_type": image.content_type,
            "size_bytes": len(raw),
        }
        return result

    except HTTPException:
        raise
    except Exception as e:
        # Avoid leaking long tracebacks to clients; keep logs in console instead.
        print(f"[estimator] estimation error: {e}")
        raise HTTPException(status_code=500, detail=f"estimation error: {e}")

# ======== How to run (examples) ========
# 1) One‑time env (Mac mini):
#    export AI_SHARED_TOKEN='your_long_random_token'
#    export AI_PRELOAD=1             # optional but recommended
#    export HF_TOKEN='...'           # if your models require HF auth
#    # Optional:
#    # export CORS_ORIGINS='http://192.168.0.61:8000'
#    # export AI_MAX_MB=8
#    # export ALLOW_WEBP=1
#
# 2) Start:
#    uvicorn estimator_service:app --host 0.0.0.0 --port 8100
#
# 3) Local test:
#    IMG="$HOME/Desktop/gymmis-ai/meal.jpg"
#    curl -v -X POST "http://127.0.0.1:8100/estimate" \
#      -H "x-api-token: $AI_SHARED_TOKEN" \
#      -F "image=@${IMG}"
#
# 4) From Ubuntu:
#    curl -s --connect-timeout 3 "http://<MAC_IP>:8100/health"
#    curl -v -X POST "http://<MAC_IP>:8100/estimate" \
#      -H "x-api-token: $AI_SHARED_TOKEN" \
#      -F "image=@/path/to/meal.jpg"
