import anthropic
import base64
import io
import json
import logging
from datetime import date
from pathlib import Path

from PIL import Image

log = logging.getLogger("calscan")

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

{filter_instruction}"""

FIX_PROMPT = """\
The JSON you returned could not be parsed. The error was: {error}

Your previous (broken) output started with:
{preview}

Return ONLY the corrected, complete, valid JSON array — no explanation, no markdown fences.
"""


def _call_claude(client: anthropic.Anthropic, messages: list[dict]) -> str:
    """Make an API call and return the text content."""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8192,
        messages=messages,
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "max_tokens":
        log.warning("Response hit max_tokens limit — output may be truncated")

    raw = next(
        (block.text for block in response.content if block.type == "text"), ""
    ).strip()

    # Strip markdown fences if Claude added them despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return raw


def _parse_with_retry(client: anthropic.Anthropic, messages: list[dict], max_retries: int = 2) -> list[dict]:
    """Call Claude, parse JSON, and retry up to max_retries times on failure."""
    raw = _call_claude(client, messages)

    for attempt in range(max_retries + 1):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            if attempt == max_retries:
                raise ValueError(
                    f"Claude returned invalid JSON after {max_retries + 1} attempt(s). "
                    f"Last error: {exc}. Raw output preview: {raw[:300]!r}"
                ) from exc

            log.warning(f"JSON parse failed (attempt {attempt + 1}): {exc} — asking Claude to fix it")

            fix_message = {
                "role": "user",
                "content": FIX_PROMPT.format(
                    error=str(exc),
                    preview=raw[:500],
                ),
            }
            # Build conversation: original exchange + Claude's bad reply + fix request
            fix_messages = messages + [
                {"role": "assistant", "content": raw},
                fix_message,
            ]
            raw = _call_claude(client, fix_messages)

    raise ValueError("Unreachable")  # satisfies type checkers


def _encode_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    media_type = SUPPORTED_TYPES.get(path.suffix.lower())

    if media_type:
        with open(image_path, "rb") as fh:
            data = base64.standard_b64encode(fh.read()).decode("utf-8")
        return data, media_type

    # Auto-convert unsupported formats (BMP, TIFF, HEIC, etc.) to PNG
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return data, "image/png"


def extract_events(image_path: str, year: int | None = None, filter_prompt: str | None = None) -> list[dict]:
    """Send a calendar image to Claude and return a list of raw event dicts.

    filter_prompt: optional natural language instruction, e.g. "only include hockey events"
    """
    client = anthropic.Anthropic()
    image_data, media_type = _encode_image(image_path)
    default_year = year or date.today().year

    filter_instruction = ""
    if filter_prompt and filter_prompt.strip():
        filter_instruction = (
            f"IMPORTANT — USER FILTER INSTRUCTION (apply this before returning results):\n"
            f"{filter_prompt.strip()}\n"
            f"Only include events that match this instruction. Exclude everything else."
        )

    prompt = EXTRACTION_PROMPT.format(year=default_year, filter_instruction=filter_instruction)

    messages = [
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
    ]

    return _parse_with_retry(client, messages)
