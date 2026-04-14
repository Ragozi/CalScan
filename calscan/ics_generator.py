import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from icalendar import Calendar, Event

from .parser import CalendarEvent


def _build_calendar(
    events: list[CalendarEvent],
    calendar_name: str = "CalScan",
    timezone: str = "UTC",
) -> Calendar:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    cal = Calendar()
    cal.add("prodid", "-//CalScan//CalScan 1.0//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("method", "PUBLISH")
    cal.add("x-wr-calname", calendar_name)
    cal.add("x-wr-timezone", timezone)

    stamp = datetime.now(tz=ZoneInfo("UTC"))

    for ev in events:
        vevent = Event()
        vevent.add("uid", str(uuid.uuid4()))
        vevent.add("summary", ev.title)
        vevent.add("dtstamp", stamp)

        if ev.all_day or ev.start_time is None:
            vevent.add("dtstart", ev.date)
            vevent.add("dtend", ev.date + timedelta(days=1))
        else:
            start_dt = datetime.combine(ev.date, ev.start_time, tzinfo=tz)
            vevent.add("dtstart", start_dt)

            if ev.end_time:
                end_dt = datetime.combine(ev.date, ev.end_time, tzinfo=tz)
            else:
                end_dt = start_dt + timedelta(hours=1)
            vevent.add("dtend", end_dt)

        desc_parts = []
        if ev.description:
            desc_parts.append(ev.description)
        if ev.recurring_note:
            desc_parts.append(f"Recurring: {ev.recurring_note}")
        if desc_parts:
            vevent.add("description", "\n".join(desc_parts))

        if ev.location:
            vevent.add("location", ev.location)

        cal.add_component(vevent)

    return cal


def generate_ics_string(
    events: list[CalendarEvent],
    calendar_name: str = "CalScan",
    timezone: str = "UTC",
) -> str:
    """Return the ICS content as a string (for API responses)."""
    cal = _build_calendar(events, calendar_name=calendar_name, timezone=timezone)
    return cal.to_ical().decode("utf-8")


def generate_ics(
    events: list[CalendarEvent],
    output_path: str,
    calendar_name: str = "CalScan",
    timezone: str = "UTC",
) -> str:
    """Write a list of CalendarEvent objects to an .ics file and return the resolved path."""
    cal = _build_calendar(events, calendar_name=calendar_name, timezone=timezone)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(cal.to_ical())
    return str(out.resolve())
