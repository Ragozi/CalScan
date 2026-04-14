import anthropic
import base64
import io
import json
from datetime import date
from pathlib import Path

from PIL import Image


SUPPORTED_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

EXTRACTION_PROMPT = """\
You are a sports and calendar schedule extraction specialist. Your task is to extract EVERY event from this image with complete accuracy.

EXTRACTION RULES:
1. Extract ALL events visible — no matter how small, faint, or crowded
2. For sports schedules: capture game/practice times, team names, Home vs Away designation, field/venue
3. For recurring events: list EACH individual occurrence as its own separate event object
4. For time zones: if a timezone is shown, include it in the description field
5. Location field: if "Home" or "Away" is indicated, include that explicitly (e.g. "Home - Lincoln Field" or "Away @ Riverside Park")
6. Handle handwritten text, printed paper schedules, screenshots, and photos of whiteboards
7. If a month/year header is visible, apply it to all events under that header
8. Common abbreviations to recognize: H=Home, A=Away, TBA/TBD=null start_time, vs/v=versus, @=at (away)
9. If a score is shown next to a past event, include it in the description field
10. Practice sessions, scrimmages, playoffs, and tournaments should all be extracted

DATE HANDLING:
- Full date shown (e.g. "Sat Jan 15"): parse completely
- Day number only (e.g. "15"): combine with the month/year header visible in the image
- No year visible: use {year}
- Always output ISO 8601 format: "YYYY-MM-DD"

TIME HANDLING:
- Convert all times to 24-hour HH:MM format
- "7:30 PM" -> "19:30", "9:00 AM" -> "09:00"
- If only start time is shown with no end time: set end_time to null
- If marked TBD/TBA: set start_time to null and all_day to true

RETURN only a valid JSON array. No markdown fences, no explanation, no prose — just the raw JSON array.

[
  {{
    "title": "string — event name, include team names if visible",
    "date": "YYYY-MM-DD",
    "start_time": "HH:MM or null",
    "end_time": "HH:MM or null",
    "location": "string or null — include Home/Away designation when shown",
    "description": "string or null — extra details, score, notes, abbreviations decoded",
    "all_day": false,
    "recurring_note": "string or null — ONLY on first occurrence: describe the pattern e.g. 'Weekly Tuesday practice'"
  }}
]

If you cannot determine the year from the image, assume {year}.
"""


def _encode_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    media_type = SUPPORTED_TYPES.get(path.suffix.lower())

    if media_type:
        with open(image_path, "rb") as fh:
            data = base64.standard_b64encode(fh.read()).decode("utf-8")
        return data, media_type

    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return data, "image/png"


def extract_events(image_path: str, year: int | None = None) -> list[dict]:
    """Send a calendar image to Claude and return a list of raw event dicts."""
    client = anthropic.Anthropic()

    image_data, media_type = _encode_image(image_path)
    default_year = year or date.today().year
    prompt = EXTRACTION_PROMPT.format(year=default_year)

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    ) as stream:
        response = stream.get_final_message()

    raw = next(
        (block.text for block in response.content if block.type == "text"), ""
    ).strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return json.loads(raw)
