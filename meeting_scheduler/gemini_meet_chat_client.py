# gemini_meet_chat_client.py  — quiet chat client for scheduling Google Meet events
import os, sys, json, asyncio
from datetime import datetime, timedelta
from dateutil import parser as dateparse

# Silence noisy libs early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from google import genai
from google.genai import types
import pytz

import scheduler_core as core  # your Google Calendar helper

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LOCAL_TZ = os.getenv("LOCAL_TZ", core.DEFAULT_TZ if hasattr(core, "DEFAULT_TZ") else "America/Chicago")

INSTRUCTIONS = """You are an assistant that extracts meeting details as JSON.
Return ONLY a JSON object (no backticks, no prose):
{
  "title": "<string, default 'Meeting'>",
  "attendees": ["email1","email2", ...],
  "start": "<RFC3339 like 2025-10-28T14:00:00-05:00 OR omit if using a time window>",
  "end": "<RFC3339 OR omit>",
  "timezone": "<IANA tz like America/Chicago>",
  "description": "<optional>",
  "duration_minutes": <integer, optional>,
  "window": { "start": "<RFC3339>", "end": "<RFC3339>" }  // optional time range
}
Rules:
- If user says “N minutes”, set duration_minutes = N.
- If window and duration exist, pick the earliest feasible start in window; set end = start + duration.
- If end missing and duration exists, set end = start + duration.
- If only a window exists, pick earliest start in window and set default duration 30 minutes.
- Always choose datetimes in the FUTURE relative to 'Current date/time' and the given timezone (or America/Chicago).
- Output JSON ONLY.
"""

def _tz(tzname: str):
    try: return pytz.timezone(tzname)
    except Exception: return pytz.timezone("America/Chicago")

def _coerce_future(dt, tzname: str):
    tz = _tz(tzname)
    now = datetime.now(tz)
    d = dt.astimezone(tz)
    # If scheduled time is in the past, move ahead in 1-week steps until future
    while d <= now:
        d = d + timedelta(weeks=1)
    return d

def _parse_iso(dt_str: str, tzname: str):
    # robust parse; add tz if missing
    tz = _tz(tzname)
    d = dateparse.parse(dt_str)
    if d.tzinfo is None:
        d = tz.localize(d)
    return d

def _get_text(resp) -> str:
    # Avoid resp.text to prevent SDK warnings
    out = []
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for p in (getattr(content, "parts", None) or []):
            if getattr(p, "text", None):
                out.append(p.text)
    return "\n".join(out).strip()

def _gen_json_or_retry(ai, model: str, prompt: str, max_output_tokens: int = 512) -> dict:
    # First: force JSON MIME
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        max_output_tokens=max_output_tokens,
        temperature=0.2,
    )
    resp = ai.models.generate_content(model=model, contents=[types.Content(role="user", parts=[types.Part(text=prompt)])], config=cfg)
    txt = _get_text(resp)

    # If empty/blocked, retry with plain request but still ask for raw JSON
    if not txt:
        cfg2 = types.GenerateContentConfig(max_output_tokens=max_output_tokens, temperature=0.2)
        resp2 = ai.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt + "\n\nReturn ONLY raw JSON (no code fences).")])],
            config=cfg2,
        )
        txt = _get_text(resp2)
        if not txt:
            # Give a compact reason if available
            fr = None
            if getattr(resp, "candidates", None):
                fr = getattr(resp.candidates[0], "finish_reason", None)
            raise SystemExit(f"Assistant didn’t produce JSON (finish_reason={fr}). Try rephrasing.")

    # Some models still wrap in fences—strip if present
    s = txt.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if "```" in s:
            s = s.split("```", 1)[0]
        s = s.strip()

    return json.loads(s)

def _summarize_event(evt: dict) -> str:
    start = evt.get("start", {})
    end = evt.get("end", {})
    st = start.get("dateTime"); et = end.get("dateTime")
    tz = start.get("timeZone") or end.get("timeZone") or LOCAL_TZ
    title = evt.get("summary") or "Meeting"
    link = evt.get("hangoutLink") or evt.get("meetLink") or evt.get("htmlLink", "")
    attendees = evt.get("attendees", [])
    if isinstance(attendees, list):
        attendees = [a.get("email", a) if isinstance(a, dict) else a for a in attendees]
    return (
        f"Scheduled **{title}**\n"
        f"- When: {st} → {et} ({tz})\n"
        f"- Attendees: {', '.join(attendees) if attendees else '—'}\n"
        f"- Meet: {link or 'created'}"
    )

def _pick_start_end(data: dict, tzname: str):
    # Choose start/end with fallbacks and future coercion
    start_iso = data.get("start")
    end_iso   = data.get("end")
    duration  = data.get("duration_minutes")
    window    = data.get("window") or {}

    if not start_iso and window.get("start"):
        start_iso = window["start"]
    if not start_iso:
        raise SystemExit("I couldn’t find a start time. Try giving a time or a window.")

    sd = _parse_iso(start_iso, tzname)
    sd = _coerce_future(sd, tzname)

    if duration and (not end_iso):
        ed = sd + timedelta(minutes=int(duration))
    elif end_iso:
        ed = _parse_iso(end_iso, tzname)
        ed = _coerce_future(ed, tzname)
        if ed <= sd:
            ed = sd + timedelta(minutes=30)
    else:
        # default 30 minutes
        ed = sd + timedelta(minutes=30)

    return sd, ed

async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY.", file=sys.stderr); sys.exit(1)

    ai = genai.Client(api_key=api_key)
    tzname = LOCAL_TZ

    print("Meeting assistant ready. Type your request (e.g., 'schedule 45 min with a@b.com tomorrow 2–5pm CT').")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            return

        now = datetime.now(_tz(tzname)).isoformat()
        prompt = INSTRUCTIONS + f"\n\nCurrent date/time: {now}\n\nUser request:\n{user}"

        # Extract JSON from Gemini (robust)
        try:
            data = _gen_json_or_retry(ai, MODEL, prompt)
        except Exception as e:
            print(f"\nAssistant: Sorry, I couldn’t parse that into a time. {e}")
            continue

        title = (data.get("title") or "Meeting").strip()
        attendees = list({(a or "").strip() for a in data.get("attendees", []) if (a or "").strip()})
        tzname = (data.get("timezone") or tzname).strip() or LOCAL_TZ
        description = data.get("description") or ""

        try:
            sd, ed = _pick_start_end(data, tzname)
        except SystemExit as e:
            print(f"\nAssistant: {e}")
            continue

        # Create the calendar event (send updates so attendees get invites)
        try:
            evt = core.create_meet_event(
                title=title,
                start=sd.isoformat(),
                end=ed.isoformat(),
                attendees=attendees,
                description=description,
                time_zone=tzname,
                send_updates="all",
            )
        except Exception as e:
            print(f"\nAssistant: I couldn’t create the event ({e}).")
            continue

        # Minimal, friendly confirmation
        print("\nAssistant:", _summarize_event(evt))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
