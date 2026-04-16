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
You are a calendar and schedule extraction specialist. Your task is to extract EVERY event from this image with complete accuracy.

STEP 1 — CLASSIFY THE IMAGE (do this silently, do not output it):
Determine which category best describes the image:
  A) SPORTS SCHEDULE — season schedule, game/practice list, team calendar, tournament bracket
  B) SINGLE EVENT — booking confirmation, appointment reminder, class confirmation, ticket, receipt with a date/time
  C) GENERAL CALENDAR — school calendar, work schedule, monthly/weekly planner, printed agenda

STEP 2 — APPLY THE RIGHT EXTRACTION LOGIC:

If Category A (SPORTS SCHEDULE):
- Extract every game, practice, scrimmage, and tournament as individual events
- Capture team names, Home vs Away designation (H/A/@/vs), field/venue
- Recognize abbreviations: H=Home, A=Away, TBA/TBD=null time, vs/v=versus, @=at (away)
- Include scores from past games in the description field
- List EACH recurring occurrence as its own separate event

If Category B (SINGLE EVENT / BOOKING CONFIRMATION):
- Extract the one event with all available details
- Title should be the service/class/appointment type (e.g. "50 Minute Stretch Session")
- Include provider, instructor, or staff name in description (e.g. "Flexologist: Jeff")
- Include the business/venue name in location
- Capture exact start and end times

If Category C (GENERAL CALENDAR):
- Extract ALL visible events — no matter how small or crowded
- Apply any visible month/year header to events that only show a day number
- Include any noted location, teacher, or organizer details

UNIVERSAL RULES (apply to all categories):
1. Never skip an event — extract everything visible
2. For recurring events: list EACH occurrence as its own separate event object
3. If a timezone is shown, include it in the description field
4. Staff, instructor, provider, or coach names belong in the description field

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

VOICE_PROMPT = """\
The user has dictated the following calendar entry by voice. Extract every event described and return them as a JSON array.

USER'S DICTATION:
{text}

EXTRACTION RULES:
1. Parse natural language dates and times — "this Saturday", "next Tuesday at 3", "every Thursday through June"
2. For recurring events: list EACH individual occurrence as its own separate event object
3. If no year is mentioned, assume {year}
4. If no end time is mentioned, set end_time to null
5. If no specific time is mentioned, set all_day to true
6. Include any person names, team names, locations, or notes in the appropriate fields
7. If the user mentions a relative day ("tomorrow", "next week"), interpret relative to today's date: {today}

Examples of what users might say:
- "Jake has soccer practice every Tuesday at 4pm at Riverside Park through June"
- "Doctor appointment Friday the 18th at 2:30"
- "Team dinner Saturday night at 7 at Carmine's, bring the whole family"

RETURN only a valid JSON array. No markdown fences, no explanation — just the raw JSON array.

[
  {{
    "title": "string — clear event name",
    "date": "YYYY-MM-DD",
    "start_time": "HH:MM or null",
    "end_time": "HH:MM or null",
    "location": "string or null",
    "description": "string or null — any extra details the user mentioned",
    "all_day": false,
    "recurring_note": "string or null — ONLY on first occurrence if recurring"
  }}
]

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
        timeout=60.0,
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


def extract_events_from_text(text: str, year: int | None = None, filter_prompt: str | None = None) -> list[dict]:
    """Parse a voice-dictated or typed calendar description into structured events.

    text: the raw dictated string from the user
    filter_prompt: optional natural language filter
    """
    client = anthropic.Anthropic()
    today = date.today()
    default_year = year or today.year

    filter_instruction = ""
    if filter_prompt and filter_prompt.strip():
        filter_instruction = (
            f"IMPORTANT — USER FILTER INSTRUCTION (apply this before returning results):\n"
            f"{filter_prompt.strip()}\n"
            f"Only include events that match this instruction. Exclude everything else."
        )

    prompt = VOICE_PROMPT.format(
        text=text.strip(),
        year=default_year,
        today=today.isoformat(),
        filter_instruction=filter_instruction,
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return _parse_with_retry(client, messages)
