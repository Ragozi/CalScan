import anthropic
import base64
import json
from datetime import date
from pathlib import Path

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

EXTRACTION_PROMPT = """You are a calendar event extraction assistant. Analyze this calendar image and extract ALL visible events.

For each event, produce a JSON object with these fields:
- title: string - event name or title
- date: string - ISO 8601 date, e.g. "2025-01-15"
- start_time: string or null - 24-hour "HH:MM", null if all-day
- end_time: string or null - 24-hour "HH:MM", null if unknown or not shown
- description: string or null - any extra details visible in the image
- location: string or null - location if visible
- all_day: boolean - true when no specific time is shown

Return ONLY a valid JSON array of event objects. No explanation, no markdown fences, no prose.

If you cannot determine the year from the image, assume the year is {year}.
If you see recurring events, include each occurrence as a separate object."""


def _encode_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("utf-8")
    return data, media_type


def extract_events(image_path: str, year: int | None = None) -> list[dict]:
    """Send a calendar image to Claude and return a list of raw event dicts."""
    client = anthropic.Anthropic()
    image_data, media_type = _encode_image(image_path)
    default_year = year or date.today().year
    prompt = EXTRACTION_PROMPT.format(year=default_year)
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    ) as stream:
        response = stream.get_final_message()
    raw = next((block.text for block in response.content if block.type == "text"), "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)
