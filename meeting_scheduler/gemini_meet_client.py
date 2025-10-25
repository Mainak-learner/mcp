# gemini_meet_client.py
import os, sys, json, argparse
import google.generativeai as genai
from dateutil import parser as dateparse
import scheduler_core as core  # reuse the same engine locally
from datetime import datetime, timedelta
import pytz


INSTRUCTIONS = """You are an assistant that extracts meeting details as JSON.

Return ONLY a JSON object:
{
  "title": "<string, default 'Meeting'>",
  "attendees": ["email1","email2", ...],
  "start": "<RFC3339 like 2025-10-28T14:00:00-05:00 OR omit if using a time window>",
  "end": "<RFC3339 OR omit>",
  "timezone": "<IANA tz like America/Chicago>",
  "description": "<optional>",
  "duration_minutes": <integer, optional>,
  "window": { "start": "<RFC3339>", "end": "<RFC3339>" }  // optional range
}
Rules:
- If user says “45 minutes”, set duration_minutes = 45.
- If a window and a duration are given, pick the earliest feasible time in the window and set end = start + duration.
- If end is missing and duration_minutes exists, set end = start + duration_minutes.
- Always choose datetimes in the FUTURE relative to now.
- Use the user’s timezone if provided; otherwise America/Chicago.
- Output JSON ONLY."""


def _tz(tzname):
    import pytz
    try: return pytz.timezone(tzname)
    except Exception: return pytz.timezone("America/Chicago")

def _coerce_future_start(start_iso: str, tzname: str):
    tzinfo = _tz(tzname)
    now = datetime.now(tzinfo)
    # robust parse (handles trailing Z)
    dt = datetime.fromisoformat(start_iso.replace("Z","+00:00"))
    if dt.tzinfo is None:
        dt = tzinfo.localize(dt)
    dt = dt.astimezone(tzinfo)
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
    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "max_output_tokens": 512,
        }
    )
    txt = resp.text.strip()

    def _strip_code_fence(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            # drop an optional leading language tag like json\n
            if s.lower().startswith("json"):
                s = s[4:] if s[:4].lower() == "json" else s
            s = s.strip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
        return s

    txt = _strip_code_fence(txt)
    data = json.loads(txt)  # unchanged


    # Safety: ensure it's JSON
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        print("Gemini did not return JSON:\n", txt, file=sys.stderr); sys.exit(1)

    title = data.get("title") or "Meeting"
    attendees = list({e.strip() for e in data.get("attendees", []) if e and e.strip()})
    tzname = (data.get("timezone") or os.environ.get("LOCAL_TZ") or "America/Chicago").strip()
    desc = data.get("description") or ""

    duration = data.get("duration_minutes")
    duration = int(duration) if isinstance(duration, int) and duration > 0 else None

    start_iso = data.get("start")
    end_iso   = data.get("end")
    window    = data.get("window") or {}

    if (not start_iso) and window.get("start") and window.get("end"):
        start_iso = window["start"]
    if not start_iso:
        raise SystemExit("No start time parsed (need a time or a window).")

    start_dt = _coerce_future_start(start_iso, tzname)
    if duration:
        end_dt = start_dt + timedelta(minutes=duration)
    elif end_iso:
        end_dt = _coerce_future_start(end_iso, tzname)
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(minutes=30)
    else:
        end_dt = start_dt + timedelta(minutes=30)

    evt = core.create_meet_event(
        title=title,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        attendees=attendees,
        description=desc,
        time_zone=tzname,
        send_updates="all"   # ensure invite emails go out
    )
    print(json.dumps(evt, indent=2))
if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("Set GOOGLE_API_KEY.", file=sys.stderr); sys.exit(1)
    main()
