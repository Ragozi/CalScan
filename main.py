"""CalScan FastAPI backend — zero data retention photo-to-calendar service."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

load_dotenv()

from calscan.ics_generator import generate_ics_string
from calscan.parser import parse_events
from calscan.vision import extract_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("calscan")

app = FastAPI(
    title="CalScan API",
    description="Upload a calendar photo and get back structured events + optional ICS.",
    version="1.0.0",
)

CALSCAN_API_KEY: str | None = os.getenv("CALSCAN_API_KEY")


def _require_api_key(x_api_key: str | None) -> None:
    if not CALSCAN_API_KEY:
        return
    if x_api_key != CALSCAN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


@app.get("/health", tags=["ops"])
def health_check() -> dict:
    return {"status": "ok", "service": "CalScan API", "version": "1.0.0"}


@app.post("/scan", tags=["scan"])
async def scan_calendar(
    photo: Annotated[UploadFile, File(description="Calendar image")],
    year: Annotated[int | None, Form()] = None,
    calendar_name: Annotated[str, Form()] = "CalScan",
    return_ics: Annotated[bool, Form()] = False,
    timezone: Annotated[str, Form()] = "UTC",
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JSONResponse:
    """Upload a calendar photo. Photo is deleted immediately after processing."""
    _require_api_key(x_api_key)

    tmp_path: str | None = None

    try:
        suffix = Path(photo.filename or "upload.jpg").suffix.lower() or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            contents = await photo.read()
            tmp.write(contents)

        log.info(f"Processing upload | filename={photo.filename} | size={len(contents)} bytes")

        raw_events = extract_events(tmp_path, year=year)
        events = parse_events(raw_events)
        log.info(f"Extracted {len(events)} event(s)")

    except Exception as exc:
        log.exception("Error during event extraction")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract events: {exc}",
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            log.info(f"PHOTO DELETED for privacy | path={tmp_path}")

    serialized = [
        {
            "title": ev.title,
            "date": ev.date.isoformat(),
            "start_time": ev.start_time.strftime("%H:%M") if ev.start_time else None,
            "end_time": ev.end_time.strftime("%H:%M") if ev.end_time else None,
            "location": ev.location,
            "description": ev.description,
            "all_day": ev.all_day,
            "recurring_note": ev.recurring_note,
        }
        for ev in events
    ]

    payload: dict = {
        "events": serialized,
        "event_count": len(serialized),
        "photo_deleted": True,
        "message": f"Successfully extracted {len(serialized)} event(s). Photo deleted immediately after processing.",
    }

    if return_ics:
        payload["ics_content"] = generate_ics_string(
            events, calendar_name=calendar_name, timezone=timezone
        )

    return JSONResponse(content=payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
