import os
import json
import base64
import tempfile
from typing import List, Optional, Dict, Any, Tuple

import requests
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

DOWNLOAD_TIMEOUT_SEC = 30


# -----------------------------
# API Models
# -----------------------------
class DetectImageRequest(BaseModel):
    postId: str
    imageUrl: str


class DetectVideoRequest(BaseModel):
    postId: str
    videoUrl: str
    sampleFps: float = 1.0
    maxFrames: int = 4


class DetectResponseObject(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    startTimeMs: int = 0
    endTimeMs: int = 0


# ✅ Option A: Rich response model
class DetectResponse(BaseModel):
    postId: str
    title: str = ""
    brand: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    keywords: List[str] = []
    suggestedCategories: List[str] = []
    notes: str = ""
    objects: List[DetectResponseObject]


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Vizucart AI", version="1.0.0")


# Root helpers
@app.get("/")
def root():
    return {"name": "Vizucart AI", "status": "ok", "docs": "/docs", "health": "/health"}


@app.head("/")
def head_root():
    return


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Helpers: Download + Encode
# -----------------------------
def _download_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "VizucartAI/1.0 (+https://vizucart.ai)",
        "Accept": "*/*",
    }
    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=DOWNLOAD_TIMEOUT_SEC,
            allow_redirects=True,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {e}")

    if resp.status_code < 200 or resp.status_code >= 300:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download URL (status={resp.status_code})",
        )

    if not resp.content:
        raise HTTPException(status_code=400, detail="Downloaded content was empty")

    return resp.content


def _bytes_to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _infer_mime_from_url(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def _jpeg_encode_bgr(frame_bgr) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame to JPEG")
    return buf.tobytes()


def _sample_video_frames_from_path(
    *,
    video_path: str,
    sample_fps: float,
    max_frames: int,
) -> Tuple[List[str], List[Dict[str, int]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_frames / fps) if total_frames > 0 else 0

    sample_every_n = max(int(round(fps / max(sample_fps, 0.1))), 1)

    frames_data_urls: List[str] = []
    time_windows: List[Dict[str, int]] = []

    frame_index = 0
    grabbed = 0

    while grabbed < max_frames:
        ok = cap.grab()
        if not ok:
            break

        if frame_index % sample_every_n == 0:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_index += 1
                continue

            jpeg_bytes = _jpeg_encode_bgr(frame)
            data_url = _bytes_to_data_url(jpeg_bytes, "image/jpeg")

            t_sec = frame_index / fps
            start_ms = int(t_sec * 1000)

            if duration_sec > 0:
                end_sec = min((t_sec + (1.0 / max(sample_fps, 0.1))), duration_sec)
                end_ms = int(end_sec * 1000)
            else:
                end_ms = start_ms

            frames_data_urls.append(data_url)
            time_windows.append({"start": start_ms, "end": end_ms})
            grabbed += 1

        frame_index += 1

    cap.release()

    if not frames_data_urls:
        raise HTTPException(status_code=400, detail="No frames sampled from video")

    return frames_data_urls, time_windows


# -----------------------------
# OpenAI helper (Option A)
# -----------------------------
def _openai_detect_metadata_and_objects_from_images(
    *,
    post_id: str,
    image_data_urls: List[str],
    time_windows_ms: Optional[List[Dict[str, int]]] = None,
) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    timing_hint = ""
    if time_windows_ms and len(time_windows_ms) == len(image_data_urls):
        timing_hint = (
            "You are seeing multiple video frames. "
            "Use the provided frame time windows to populate startTimeMs/endTimeMs."
        )

    system_text = (
        "You are Vizucart's shopping metadata assistant. "
        "Identify shoppable items and produce product-friendly metadata. "
        "Return ONLY valid JSON (no markdown)."
    )

    # ✅ NEW schema includes: title/brand/confidence/keywords/suggestedCategories/notes/objects
    user_text = (
        f"PostId: {post_id}\n"
        f"{timing_hint}\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "title": string,\n'
        '  "brand": string,\n'
        '  "confidence": number (0 to 1),\n'
        '  "keywords": [string],\n'
        '  "suggestedCategories": [string],\n'
        '  "notes": string,\n'
        '  "objects": [\n'
        "    {\n"
        '      "label": string,\n'
        '      "confidence": number (0 to 1),\n'
        '      "startTimeMs": integer,\n'
        '      "endTimeMs": integer\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- title: a shopper-friendly primary item name (singular), e.g. 'milk frother', 'hoodie', 'tripod'.\n"
        "- brand: ONLY if clearly visible (logo/text/packaging). Otherwise return empty string \"\".\n"
        "- confidence: confidence for the title (0..1). If unsure, set lower.\n"
        "- keywords: 5-15 helpful search terms (include object labels + context like 'kitchen', 'camera', etc.).\n"
        "- suggestedCategories: 1-5 ecommerce categories, broad-to-specific.\n"
        "- notes: short helpful note; mention uncertainty if brand/model not visible.\n"
        "- objects: tangible shoppable items ONLY (avoid people/body parts).\n"
        "- If timing is unknown (single image), set startTimeMs/endTimeMs to 0.\n"
    )

    content_parts: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]

    for i, data_url in enumerate(image_data_urls):
        if time_windows_ms and i < len(time_windows_ms):
            start = time_windows_ms[i]["start"]
            end = time_windows_ms[i]["end"]
            content_parts.append(
                {"type": "input_text", "text": f"Frame {i+1} timeWindowMs: start={start}, end={end}"}
            )
        content_parts.append({"type": "input_image", "image_url": data_url})

    payload = {
        "model": OPENAI_MODEL,
        "instructions": system_text,
        "input": [{"role": "user", "content": content_parts}],
        "text": {"format": {"type": "json_object"}},
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            OPENAI_RESPONSES_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

    if resp.status_code < 200 or resp.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {resp.status_code} {resp.text}")

    data = resp.json()

    output_text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    output_text += c.get("text", "")

    if not output_text.strip():
        raise HTTPException(status_code=502, detail="OpenAI returned empty output_text")

    try:
        parsed = json.loads(output_text)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse OpenAI JSON: {e}. Raw: {output_text[:800]}",
        )

    # ---- Extract rich fields safely ----
    title = str(parsed.get("title", "") or "").strip()
    brand = str(parsed.get("brand", "") or "").strip()
    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    keywords_raw = parsed.get("keywords", []) or []
    keywords = []
    if isinstance(keywords_raw, list):
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]

    suggested_raw = parsed.get("suggestedCategories", []) or []
    suggested = []
    if isinstance(suggested_raw, list):
        suggested = [str(s).strip() for s in suggested_raw if str(s).strip()]

    notes = str(parsed.get("notes", "") or "").strip()

    # ---- Objects ----
    objs = parsed.get("objects", []) or []
    objects_list: List[DetectResponseObject] = []
    if isinstance(objs, list):
        for o in objs:
            try:
                objects_list.append(
                    DetectResponseObject(
                        label=str(o.get("label", "")).strip(),
                        confidence=float(o.get("confidence", 0.0)),
                        startTimeMs=int(o.get("startTimeMs", 0)),
                        endTimeMs=int(o.get("endTimeMs", 0)),
                    )
                )
            except Exception:
                continue

    # Filter empty labels
    objects_list = [r for r in objects_list if r.label]

    # If model forgot to set title, derive from top object
    if not title and objects_list:
        objects_sorted = sorted(objects_list, key=lambda x: x.confidence, reverse=True)
        title = objects_sorted[0].label

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    if not notes:
        if brand:
            notes = "Detected item and brand from visible packaging/logo."
        else:
            notes = "Brand/model not clearly visible. Confirm during product match."

    return {
        "title": title,
        "brand": brand,
        "confidence": confidence,
        "keywords": keywords,
        "suggestedCategories": suggested,
        "notes": notes,
        "objects": objects_list,
    }


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/v1/detect/image", response_model=DetectResponse)
def detect_image(req: DetectImageRequest):
    image_bytes = _download_bytes(req.imageUrl)
    mime = _infer_mime_from_url(req.imageUrl)
    data_url = _bytes_to_data_url(image_bytes, mime)

    result = _openai_detect_metadata_and_objects_from_images(
        post_id=req.postId,
        image_data_urls=[data_url],
        time_windows_ms=None,
    )

    return DetectResponse(postId=req.postId, **result)


@app.post("/v1/detect/image-upload", response_model=DetectResponse)
async def detect_image_upload(postId: str, file: UploadFile = File(...)):
    if not file.content_type or "image" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Upload must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image was empty")

    data_url = _bytes_to_data_url(image_bytes, file.content_type)

    result = _openai_detect_metadata_and_objects_from_images(
        post_id=postId,
        image_data_urls=[data_url],
        time_windows_ms=None,
    )

    return DetectResponse(postId=postId, **result)


@app.post("/v1/detect/video", response_model=DetectResponse)
def detect_video(req: DetectVideoRequest):
    video_bytes = _download_bytes(req.videoUrl)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        frames_data_urls, time_windows = _sample_video_frames_from_path(
            video_path=tmp.name,
            sample_fps=req.sampleFps,
            max_frames=req.maxFrames,
        )

    result = _openai_detect_metadata_and_objects_from_images(
        post_id=req.postId,
        image_data_urls=frames_data_urls,
        time_windows_ms=time_windows,
    )

    return DetectResponse(postId=req.postId, **result)


@app.post("/v1/detect/video-upload", response_model=DetectResponse)
async def detect_video_upload(
    postId: str,
    file: UploadFile = File(...),
    sampleFps: float = 1.0,
    maxFrames: int = 4,
):
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded video was empty")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        frames_data_urls, time_windows = _sample_video_frames_from_path(
            video_path=tmp.name,
            sample_fps=sampleFps,
            max_frames=maxFrames,
        )

    result = _openai_detect_metadata_and_objects_from_images(
        post_id=postId,
        image_data_urls=frames_data_urls,
        time_windows_ms=time_windows,
    )

    return DetectResponse(postId=postId, **result)
