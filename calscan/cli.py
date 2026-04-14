import sys
from pathlib import Path

import click

from .ics_generator import generate_ics
from .parser import parse_events
from .vision import extract_events


@click.command()
@click.argument("image_path", type=click.Path(exists=True, readable=True))
@click.option("--output", "-o", default=None, help="Output .ics file path. Defaults to <image-stem>.ics in the current directory.")
@click.option("--year", "-y", default=None, type=int, help="Default year to assume when the image does not show one (default: current year).")
@click.option("--verbose", "-v", is_flag=True, help="Print each extracted event before writing the file.")
def main(image_path: str, output: str | None, year: int | None, verbose: bool) -> None:
    """CalScan: extract calendar events from an image and export to a .ics file.

    IMAGE_PATH is the path to the calendar image (JPEG, PNG, WEBP, or GIF).
    Requires ANTHROPIC_API_KEY to be set in the environment.
    """
    if output is None:
        output = Path(image_path).stem + ".ics"
    click.echo(f"Scanning {image_path} ...")
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
            click.echo(f"  * {ev.date}{time_str}  {ev.title}")
    else:
        click.echo(f"Extracted {len(events)} event(s).")
    try:
        saved_path = generate_ics(events, output)
        click.echo(f"Saved: {saved_path}")
    except Exception as exc:
        click.echo(f"Error writing .ics file: {exc}", err=True)
        sys.exit(1)
