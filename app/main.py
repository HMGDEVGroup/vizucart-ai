import os
import json
import base64
import tempfile
from typing import List, Optional, Dict, Any, Tuple

import requests
import cv2

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# Config
# -----------------------------
DOWNLOAD_TIMEOUT_SEC = 15

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Vizucart AI API", version="1.0.0")

# Allow everything for dev (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Root + Health Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "vizucart-ai"}

# ✅ ADD THIS: fixes Render/Cloudflare HEAD checks
@app.head("/")
def head_root():
    return


# -----------------------------
# Request / Response Models
# -----------------------------
class DetectImageRequest(BaseModel):
    postId: str = Field(..., description="Client post ID")
    imageUrl: str = Field(..., description="Public image URL")


class DetectVideoRequest(BaseModel):
    postId: str = Field(..., description="Client post ID")
    videoUrl: str = Field(..., description="Public video URL")
    sampleFps: float = Field(1.0, description="Frames per second to sample")
    maxFrames: int = Field(4, description="Max number of frames to analyze")


class DetectResponseObject(BaseModel):
    label: str
    confidence: float
    startTimeMs: Optional[int] = None
    endTimeMs: Optional[int] = None


class DetectResponse(BaseModel):
    postId: str
    objects: List[DetectResponseObject]


# -----------------------------
# Helpers
# -----------------------------
def _download_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "vizucart-ai/1.0",
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
    # OpenAI supports base64 data URLs
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _infer_mime_from_url(url: str) -> str:
    u = url.lower()
    if u.endswith(".png"):
        return "image/png"
    if u.endswith(".webp"):
        return "image/webp"
    if u.endswith(".gif"):
        return "image/gif"
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
    """
    Returns:
      frames_data_urls: list of base64 data URLs (JPEG frames)
      time_windows: list of {"start":ms,"end":ms} for each frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_frames / fps) if total_frames > 0 else 0

    # We want ~sample_fps frames per second, but cap at max_frames
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

            # Time window for this sampled frame
            t_sec = frame_index / fps
            start_ms = int(t_sec * 1000)

            if duration_sec > 0:
                end_sec = min(
                    (t_sec + (1.0 / max(sample_fps, 0.1))),
                    duration_sec,
                )
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
# OpenAI Call
# -----------------------------
def _openai_detect_objects_from_images(
    *,
    post_id: str,
    image_data_urls: List[str],
    time_windows_ms: Optional[List[Dict[str, int]]] = None,
) -> List[DetectResponseObject]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    timing_hint = ""
    if time_windows_ms and len(time_windows_ms) == len(image_data_urls):
        timing_hint = (
            "For each object, include startTimeMs and endTimeMs matching the frame timeWindowMs.\n"
        )

    system_prompt = (
        "You are a precise computer vision assistant.\n"
        "Return ONLY strict JSON with this schema:\n"
        "{\n"
        '  "objects": [\n'
        "    {\n"
        '      "label": string,\n'
        '      "confidence": number,\n'
        '      "startTimeMs": number|null,\n'
        '      "endTimeMs": number|null\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- labels must be short, lower-case.\n"
        "- confidence must be 0..1.\n"
        "- avoid people/body parts.\n"
        + timing_hint
    )

    user_text = (
        "Detect the most important shopping/product objects in these images.\n"
        "If timeWindowMs hints are provided, attach the timing fields.\n"
    )

    content_parts: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]

    for i, data_url in enumerate(image_data_urls):
        if time_windows_ms and i < len(time_windows_ms):
            start = time_windows_ms[i]["start"]
            end = time_windows_ms[i]["end"]
            content_parts.append(
                {
                    "type": "input_text",
                    "text": f"Frame {i+1} timeWindowMs: start={start}, end={end}",
                }
            )

        content_parts.append({"type": "input_image", "image_url": data_url})

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": content_parts,
            },
        ],
        "max_output_tokens": 500,
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
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI error: {resp.status_code} {resp.text}",
        )

    data = resp.json()

    # Extract output text from Responses API
    output_text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    output_text += part.get("text", "")

    if not output_text.strip():
        raise HTTPException(status_code=502, detail="No output_text returned from OpenAI")

    try:
        parsed = json.loads(output_text)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse OpenAI JSON: {e}. Raw: {output_text[:800]}",
        )

    objs = parsed.get("objects", [])
    result: List[DetectResponseObject] = []

    for o in objs:
        try:
            result.append(
                DetectResponseObject(
                    label=str(o.get("label", "")).strip(),
                    confidence=float(o.get("confidence", 0)),
                    startTimeMs=o.get("startTimeMs", None),
                    endTimeMs=o.get("endTimeMs", None),
                )
            )
        except Exception:
            continue

    # Remove empty labels
    result = [r for r in result if r.label]

    return result


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/v1/detect/image", response_model=DetectResponse)
def detect_image(req: DetectImageRequest):
    image_bytes = _download_bytes(req.imageUrl)
    mime = _infer_mime_from_url(req.imageUrl)
    data_url = _bytes_to_data_url(image_bytes, mime)

    objects = _openai_detect_objects_from_images(
        post_id=req.postId,
        image_data_urls=[data_url],
        time_windows_ms=None,
    )
    return DetectResponse(postId=req.postId, objects=objects)


@app.post("/v1/detect/image-upload", response_model=DetectResponse)
async def detect_image_upload(postId: str, file: UploadFile = File(...)):
    if file.content_type and "image" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image was empty")

    data_url = _bytes_to_data_url(image_bytes, "image/jpeg")

    objects = _openai_detect_objects_from_images(
        post_id=postId,
        image_data_urls=[data_url],
        time_windows_ms=None,
    )
    return DetectResponse(postId=postId, objects=objects)


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

    objects = _openai_detect_objects_from_images(
        post_id=req.postId,
        image_data_urls=frames_data_urls,
        time_windows_ms=time_windows,
    )

    return DetectResponse(postId=req.postId, objects=objects)


@app.post("/v1/detect/video-upload", response_model=DetectResponse)
async def detect_video_upload(
    postId: str,
    file: UploadFile = File(...),
    sampleFps: float = 1.0,
    maxFrames: int = 4,
):
    if file.content_type and "video" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Upload must be a video file")

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

    objects = _openai_detect_objects_from_images(
        post_id=postId,
        image_data_urls=frames_data_urls,
        time_windows_ms=time_windows,
    )

    return DetectResponse(postId=postId, objects=objects)
