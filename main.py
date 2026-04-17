"""CalScan FastAPI backend — zero data retention photo-to-calendar service."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from calscan.ics_generator import generate_ics_string
from calscan.parser import parse_events
from calscan.usage import UPGRADE_URL, get_usage, increment_usage
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


def _require_api_key(x_api_key: str | None) -> None:
    """If CALSCAN_API_KEY is set in env, enforce it. Otherwise open for dev."""
    if not CALSCAN_API_KEY:
        return
    if x_api_key != CALSCAN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


def _error(message: str, code: int = 422) -> JSONResponse:
    """Consistent error envelope so clients always get the same shape."""
    log.error(message)
    return JSONResponse(
        status_code=code,
        content={"success": False, "message": message},
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health_check() -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    supabase_url = os.getenv("SUPABASE_URL", "")
    return {
        "status": "ok",
        "service": "CalScan API",
        "version": "2.0.0",
        "anthropic_key_set": bool(api_key),
        "anthropic_key_prefix": api_key[:14] + "..." if api_key else "MISSING",
        "supabase_connected": bool(supabase_url),
    }


@app.get("/debug-env", tags=["ops"])
def debug_env() -> dict:
    """Lists which expected env vars are present (no values exposed)."""
    keys_to_check = [
        "ANTHROPIC_API_KEY", "CALSCAN_API_KEY", "PORT",
        "RAILWAY_ENVIRONMENT", "SUPABASE_URL", "SUPABASE_SERVICE_KEY",
    ]
    return {k: bool(os.getenv(k)) for k in keys_to_check}


# ---------------------------------------------------------------------------
# POST /check_usage  — check a user's scan quota
# ---------------------------------------------------------------------------

class CheckUsageRequest(BaseModel):
    user_id: str


@app.post("/check_usage", tags=["usage"])
async def check_usage(
    body: CheckUsageRequest,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JSONResponse:
    """Check how many scans a user has remaining this month.

    Returns tier, remaining scans, and whether they can scan.
    Free tier: 8 scans/month, resets on the 1st.
    Pro tier: unlimited.
    """
    _require_api_key(x_api_key)

    if not body.user_id or not body.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="user_id is required.",
        )

    usage = get_usage(body.user_id)
    log.info(
        "Usage check | user_id=%s | tier=%s | can_scan=%s | remaining=%s",
        body.user_id, usage["tier"], usage["can_scan"], usage["scans_remaining"],
    )

    return JSONResponse(content={
        "can_scan": usage["can_scan"],
        "scans_remaining": usage["scans_remaining"],
        "tier": usage["tier"],
        "message": usage["message"],
        "upgrade_url": UPGRADE_URL if not usage["can_scan"] else None,
    })


# ---------------------------------------------------------------------------
# POST /scan
# ---------------------------------------------------------------------------

@app.post("/scan", tags=["scan"])
async def scan_calendar(
    photo: Annotated[UploadFile, File(description="Calendar image (JPEG, PNG, WEBP, GIF, BMP, TIFF)")],
    filter: Annotated[str | None, Form(description='Natural language filter, e.g. "only hockey games"')] = None,
    timezone: Annotated[str, Form(description='IANA timezone, e.g. "America/New_York"')] = "UTC",
    return_ics: Annotated[bool, Form(description="Include ics_content string in response")] = True,
    user_id: Annotated[str | None, Form(description="Supabase user ID for usage tracking")] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JSONResponse:
    """Scan a calendar photo and return structured events.

    - Photo is deleted immediately after processing (zero data retention).
    - Pass user_id to enforce per-user scan quotas (free tier: 8/month).
    - Use `filter` to limit results: "only soccer games", "just events for Jake", etc.
    - Set `return_ics=true` to get a ready-to-download ICS string in one shot.
    """
    _require_api_key(x_api_key)

    # Usage enforcement — only if user_id provided
    if user_id and user_id.strip():
        usage = get_usage(user_id)
        if not usage["can_scan"]:
            log.warning("Scan blocked | user_id=%s | reason=%s", user_id, usage["message"])
            return JSONResponse(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                content={
                    "success": False,
                    "error": "upgrade_needed",
                    "message": usage["message"],
                    "upgrade_url": UPGRADE_URL,
                },
            )

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

    # Increment usage after successful scan (free tier only)
    if user_id and user_id.strip():
        increment_usage(user_id)

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
# POST /build-ics  — accepts a user-curated event list, returns ICS
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
# POST /voice-scan  — parses a voice-dictated or typed description into events
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

    # Usage enforcement
    if body.user_id and body.user_id.strip():
        usage = get_usage(body.user_id)
        if not usage["can_scan"]:
            log.warning("Voice scan blocked | user_id=%s", body.user_id)
            return JSONResponse(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                content={
                    "success": False,
                    "error": "upgrade_needed",
                    "message": usage["message"],
                    "upgrade_url": UPGRADE_URL,
                },
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

    # Increment usage after successful voice scan
    if body.user_id and body.user_id.strip():
        increment_usage(body.user_id)

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
