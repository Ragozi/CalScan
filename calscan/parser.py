from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional

from dateutil import parser as dateutil_parser


@dataclass
class CalendarEvent:
    title: str
    date: date
    start_time: Optional[time]
    end_time: Optional[time]
    description: Optional[str]
    location: Optional[str]
    all_day: bool


def _parse_time(value: str | None) -> Optional[time]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError:
        return None


def parse_events(raw_events: list[dict]) -> list[CalendarEvent]:
    """Convert raw dicts from the vision step into CalendarEvent objects."""
    events: list[CalendarEvent] = []

    for raw in raw_events:
        try:
            event_date = dateutil_parser.parse(raw["date"]).date()
        except (KeyError, ValueError) as exc:
            print(f"Warning: skipping event with unparseable date — {raw}: {exc}")
            continue

        start_time = _parse_time(raw.get("start_time"))
        end_time = _parse_time(raw.get("end_time"))
        all_day = raw.get("all_day", start_time is None)

        events.append(
            CalendarEvent(
                title=raw.get("title") or "Untitled Event",
                date=event_date,
                start_time=start_time,
                end_time=end_time,
                description=raw.get("description") or None,
                location=raw.get("location") or None,
                all_day=all_day,
            )
        )

    return events
