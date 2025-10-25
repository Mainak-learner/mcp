# gemini_meet_client.py
import os, sys, json, argparse
import google.generativeai as genai
from dateutil import parser as dateparse
import scheduler_core as core  # reuse the same engine locally
from datetime import datetime, timedelta
import pytz


INSTRUCTIONS = """You are an assistant that extracts meeting details as JSON.

Return ONLY a JSON object with these fields:
{
  "title": "<string, default 'Meeting'>",
  "attendees": ["email1","email2", ...],
  "start": "<RFC3339 like 2025-10-28T14:00:00-05:00 OR omit if using a time window>",
  "end": "<RFC3339 OR omit>",
  "timezone": "<IANA tz like America/Chicago>",
  "description": "<optional>",
  "duration_minutes": <integer, optional>,
  "window": { "start": "<RFC3339>", "end": "<RFC3339>" }  // optional; use if user gives a range like 'between 2–5pm'
}

Rules:
- If the user gives a duration (e.g., '45 minutes'), set duration_minutes.
- If both a window and a duration are given, pick the earliest feasible time inside the window and set end = start + duration.
- If end is missing and duration_minutes is given, set end = start + duration_minutes.
- Always choose datetimes in the FUTURE relative to now.
- Use the timezone provided by the user; otherwise use America/Chicago.
- Output JSON ONLY (no backticks, no extra text).
"""

def _tz(tzname):
    try:
        return pytz.timezone(tzname)
    except Exception:
        return pytz.timezone("America/Chicago")

def _coerce_future_start(start_iso: str, tzname: str):
    """If Gemini returned a past date (wrong year), roll forward to the next week on the same weekday/time."""
    tzinfo = _tz(tzname)
    now = datetime.now(tzinfo)
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z","+00:00"))
    except Exception:
        # last resort: let dateutil handle it (but you tightened Gemini to RFC3339)
        from dateutil import parser as dp
        dt = dp.parse(start_iso)
    if dt.tzinfo is None:
        dt = tzinfo.localize(dt)
    dt = dt.astimezone(tzinfo)
    # If it's already in the future, keep it
    if dt > now:
        return dt
    # Otherwise, bump by weeks until future
    while dt <= now:
        dt = dt + timedelta(weeks=1)
    return dt


def _pick_model(requested: str | None):
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    avail = {m.name: set(m.supported_generation_methods or []) for m in genai.list_models()}
    preferred = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-flash-latest"]
    for m in ([requested] if requested else []) + preferred + list(avail.keys()):
        if m and m in avail and "generateContent" in avail[m]:
            return m
    raise RuntimeError("No suitable Gemini model available.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", help="Natural-language meeting request")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    model_name = _pick_model(args.model)
    # BEFORE (you probably had an auto-picker that overrode your choice)
    # model = genai.GenerativeModel(args.model)

    # AFTER (force exact model name)
    model_name = args.model or "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)
    
    prompt = INSTRUCTIONS + "\n\nUser request:\n" + args.request
    resp = model.generate_content(prompt)
    txt = resp.text.strip()

    # Safety: ensure it's JSON
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        print("Gemini did not return JSON:\n", txt, file=sys.stderr); sys.exit(1)

    title = data.get("title") or "Meeting"
    attendees = list({e.strip() for e in data.get("attendees", []) if e and e.strip()})
    tzname = (data.get("timezone") or os.environ.get("LOCAL_TZ") or "America/Chicago").strip()
    desc = data.get("description") or ""

    duration = None
    if isinstance(data.get("duration_minutes"), int) and data["duration_minutes"] > 0:
        duration = int(data["duration_minutes"])

    start_iso = data.get("start")
    end_iso   = data.get("end")
    window    = data.get("window") or {}

    # If user gave a time window and a duration, choose earliest slot inside window.
    if (not start_iso) and window.get("start") and window.get("end") and duration:
        # pick the window start, then we'll enforce future & compute end
        start_iso = window["start"]

    if not start_iso:
        raise SystemExit("No start time parsed. Provide a specific time or a time window plus duration.")

    # Enforce FUTURE start (fixes 2024 vs 2025)
    start_dt = _coerce_future_start(start_iso, tzname)

    # Compute end: duration wins; else use provided end; else default 30m
    if duration:
        end_dt = start_dt + timedelta(minutes=duration)
    elif end_iso:
        end_dt = _coerce_future_start(end_iso, tzname)
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(minutes=30)
    else:
        end_dt = start_dt + timedelta(minutes=30)

    # Call scheduler (ensure attendees get emails)
    evt = core.create_meet_event(
        title=title,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        attendees=attendees,
        description=desc,
        time_zone=tzname,
        send_updates="all"  # ensure notifications are sent
    )
    print(json.dumps(evt, indent=2))

if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("Set GOOGLE_API_KEY.", file=sys.stderr); sys.exit(1)
    main()
