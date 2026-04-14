import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from .ics_generator import generate_ics
from .parser import parse_events
from .vision import extract_events


@click.command()
@click.argument("image_path", type=click.Path(exists=True, readable=True))
@click.option("--output", "-o", default=None, help="Output .ics file path.")
@click.option("--year", "-y", default=None, type=int, help="Default year if not visible in image.")
@click.option("--timezone", "-t", default="UTC", help="Timezone, e.g. America/New_York.")
@click.option("--verbose", "-v", is_flag=True, help="Print each extracted event.")
def main(image_path, output, year, timezone, verbose):
    """CalScan — extract calendar events from an image and export to .ics.

    \b
    IMAGE_PATH  Path to the calendar image (JPEG, PNG, WEBP, GIF, BMP, TIFF, etc.).
    Requires ANTHROPIC_API_KEY in environment or .env file.
    """
    if output is None:
        output = Path(image_path).stem + ".ics"

    click.echo(f"Scanning {image_path} …")

    try:
        raw_events = extract_events(image_path, year=year)
    except Exception as exc:
        click.echo(f"Error during vision extraction: {exc}", err=True)
        sys.exit(1)

    events = parse_events(raw_events)

    if not events:
        click.echo("No events could be extracted from the image.", err=True)
        sys.exit(1)

    if verbose:
        click.echo(f"\nExtracted {len(events)} event(s):")
        for ev in events:
            time_str = ""
            if ev.start_time:
                time_str = f"  {ev.start_time.strftime('%H:%M')}"
                if ev.end_time:
                    time_str += f"-{ev.end_time.strftime('%H:%M')}"
            loc_str = f"  @ {ev.location}" if ev.location else ""
            click.echo(f"  . {ev.date}{time_str}  {ev.title}{loc_str}")
    else:
        click.echo(f"Extracted {len(events)} event(s).")

    try:
        saved_path = generate_ics(events, output, timezone=timezone)
        click.echo(f"Saved: {saved_path}")
    except Exception as exc:
        click.echo(f"Error writing .ics file: {exc}", err=True)
        sys.exit(1)
