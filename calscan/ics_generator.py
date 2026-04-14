import uuid
from datetime import datetime, timedelta
from pathlib import Path

from icalendar import Calendar, Event

from .parser import CalendarEvent


def generate_ics(events: list[CalendarEvent], output_path: str) -> str:
    """Write a list of CalendarEvent objects to an .ics file and return the resolved path."""
    cal = Calendar()
    cal.add("prodid", "-//CalScan//CalScan 1.0//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("method", "PUBLISH")

    stamp = datetime.utcnow()

    for ev in events:
        vevent = Event()
        vevent.add("uid", str(uuid.uuid4()))
        vevent.add("summary", ev.title)
        vevent.add("dtstamp", stamp)

        if ev.all_day or ev.start_time is None:
            vevent.add("dtstart", ev.date)
            vevent.add("dtend", ev.date + timedelta(days=1))
        else:
            start_dt = datetime.combine(ev.date, ev.start_time)
            vevent.add("dtstart", start_dt)

            if ev.end_time:
                end_dt = datetime.combine(ev.date, ev.end_time)
            else:
                end_dt = start_dt + timedelta(hours=1)
            vevent.add("dtend", end_dt)

        if ev.description:
            vevent.add("description", ev.description)
        if ev.location:
            vevent.add("location", ev.location)

        cal.add_component(vevent)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(cal.to_ical())
    return str(out.resolve())
