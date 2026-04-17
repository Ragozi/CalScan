"""CalScan FastAPI backend — zero data retention photo-to-calendar service."""

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

load_dotenv()

from calscan.ics_generator import generate_ics_string
from calscan.parser import parse_events
from calscan.vision import extract_events, extract_events_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("calscan")

app = FastAPI(
    title="CalScan API",
    description="Upload a calendar photo and get back structured events + optional ICS.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

PORT = int(os.getenv("PORT", 8080))
CALSCAN_API_KEY: str | None = os.getenv("CALSCAN_API_KEY")

# In-memory ICS store: {uuid: (ics_content, expires_at)}
# Entries expire after 5 minutes. Zero calendar data is persisted to disk.
_ics_store: dict[str, tuple[str, float]] = {}
ICS_TTL_SECONDS = 300  # 5 minutes


def _purge_expired_ics() -> None:
    """Remove expired entries from the in-memory ICS store."""
    now = time.time()
    expired = [k for k, (_, exp) in _ics_store.items() if exp < now]
    for k in expired:
        del _ics_store[k]


def _require_api_key(x_api_key: str | None) -> None:
    if not CALSCAN_API_KEY:
        return
    if x_api_key != CALSCAN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health_check() -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    return {
        "status": "ok",
        "service": "CalScan API",
        "version": "2.0.0",
        "anthropic_key_set": bool(api_key),
        "anthropic_key_prefix": api_key[:14] + "..." if api_key else "MISSING",
    }


@app.get("/debug-env", tags=["ops"])
def debug_env() -> dict:
    keys_to_check = ["ANTHROPIC_API_KEY", "CALSCAN_API_KEY", "PORT", "RAILWAY_ENVIRONMENT"]
    return {k: bool(os.getenv(k)) for k in keys_to_check}


# ---------------------------------------------------------------------------
# POST /scan
# ---------------------------------------------------------------------------

@app.post("/scan", tags=["scan"])
async def scan_calendar(
    photo: Annotated[UploadFile, File(description="Calendar image (JPEG, PNG, WEBP, GIF, BMP, TIFF)")],
    filter: Annotated[str | None, Form(description='Natural language filter, e.g. "only hockey games"')] = None,
    timezone: Annotated[str, Form(description='IANA timezone, e.g. "America/New_York"')] = "UTC",
    return_ics: Annotated[bool, Form(description="Include ics_content string in response")] = True,
    user_id: Annotated[str | None, Form(description="Supabase user ID (logged for analytics)")] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JSONResponse:
    """Scan a calendar photo and return structured events.

    Usage enforcement is handled upstream by Supabase edge functions.
    This endpoint trusts that the caller has already been authorized.
    Photo is deleted immediately after processing (zero data retention).
    """
    _require_api_key(x_api_key)

    tmp_path: str | None = None

    try:
        suffix = Path(photo.filename or "upload.jpg").suffix.lower() or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            contents = await photo.read()
            tmp.write(contents)

        log.info(
            "Scan request | file=%s | size=%d bytes | filter=%r | tz=%s | return_ics=%s | user_id=%s",
            photo.filename, len(contents), filter, timezone, return_ics, user_id,
        )

        raw_events = extract_events(tmp_path, filter_prompt=filter)
        events = parse_events(raw_events)
        log.info("Extracted %d event(s)", len(events))

    except Exception as exc:
        log.exception("Extraction failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract events: {exc}",
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            log.info("PHOTO DELETED | path=%s", tmp_path)

    serialized = [
        {
            "title": ev.title,
            "date": ev.date.isoformat(),
            "start_time": ev.start_time.strftime("%H:%M") if ev.start_time else None,
            "end_time": ev.end_time.strftime("%H:%M") if ev.end_time else None,
            "location": ev.location,
            "description": ev.description,
            "all_day": ev.all_day,
        }
        for ev in events
    ]

    payload: dict = {
        "success": True,
        "events": serialized,
        "message": f"Found {len(serialized)} event(s). Photo deleted.",
    }

    if return_ics:
        payload["ics_content"] = generate_ics_string(events, timezone=timezone)

    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# POST /build-ics
# ---------------------------------------------------------------------------

class BuildIcsRequest(BaseModel):
    events: list[dict]
    timezone: str = "UTC"
    calendar_name: str = "CalScan"


@app.post("/build-ics", tags=["scan"])
async def build_ics(
    body: BuildIcsRequest,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JSONResponse:
    """Convert a user-curated event list into a downloadable ICS string."""
    _require_api_key(x_api_key)

    events = parse_events(body.events)
    if not events:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid events in request.",
        )

    ics_content = generate_ics_string(
        events, calendar_name=body.calendar_name, timezone=body.timezone
    )
    log.info("Built ICS | %d event(s) | tz=%s", len(events), body.timezone)

    return JSONResponse(content={
        "success": True,
        "ics_content": ics_content,
        "event_count": len(events),
    })


# ---------------------------------------------------------------------------
# POST /voice-scan
# ---------------------------------------------------------------------------

class VoiceScanRequest(BaseModel):
    text: str
    timezone: str = "UTC"
    return_ics: bool = True
    filter: str | None = None
    user_id: str | None = None


@app.post("/voice-scan", tags=["scan"])
async def voice_scan(
    body: VoiceScanRequest,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JSONResponse:
    """Parse a voice-dictated or typed calendar description into structured events."""
    _require_api_key(x_api_key)

    if not body.text or not body.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="text field is required and cannot be empty.",
        )

    log.info(
        "Voice scan request | text_len=%d | filter=%r | tz=%s | user_id=%s",
        len(body.text), body.filter, body.timezone, body.user_id,
    )

    try:
        raw_events = extract_events_from_text(body.text, filter_prompt=body.filter)
        events = parse_events(raw_events)
        log.info("Extracted %d event(s) from voice input", len(events))
    except Exception as exc:
        log.exception("Voice extraction failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract events: {exc}",
        ) from exc

    serialized = [
        {
            "title": ev.title,
            "date": ev.date.isoformat(),
            "start_time": ev.start_time.strftime("%H:%M") if ev.start_time else None,
            "end_time": ev.end_time.strftime("%H:%M") if ev.end_time else None,
            "location": ev.location,
            "description": ev.description,
            "all_day": ev.all_day,
        }
        for ev in events
    ]

    payload: dict = {
        "success": True,
        "events": serialized,
        "message": f"Found {len(serialized)} event(s) from voice input.",
    }

    if body.return_ics:
        payload["ics_content"] = generate_ics_string(events, timezone=body.timezone)

    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# POST /store-ics  +  GET /ics/{token}
# Temporary ICS hosting so Android can open webcal:// links instead of
# downloading a blob/data-URI. Data lives only in RAM, expires in 5 min.
# ---------------------------------------------------------------------------

class StoreIcsRequest(BaseModel):
    ics_content: str


@app.post("/store-ics", tags=["scan"])
async def store_ics(
    body: StoreIcsRequest,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JSONResponse:
    """Store ICS content in memory for 5 minutes and return a serving URL."""
    _require_api_key(x_api_key)

    if not body.ics_content or not body.ics_content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ics_content is required.",
        )

    _purge_expired_ics()

    token = uuid.uuid4().hex
    _ics_store[token] = (body.ics_content, time.time() + ICS_TTL_SECONDS)
    log.info("Stored ICS | token=%s | expires_in=%ds", token, ICS_TTL_SECONDS)

    return JSONResponse(content={"token": token})


@app.get("/ics/{token}", tags=["scan"])
def serve_ics(token: str) -> Response:
    """Serve a previously stored ICS file by token. No auth required (token is the secret)."""
    _purge_expired_ics()

    entry = _ics_store.get(token)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ICS not found or expired.",
        )

    ics_content, _ = entry
    return Response(
        content=ics_content,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="calscan.ics"'},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
