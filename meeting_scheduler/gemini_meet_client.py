# gemini_meet_client.py
import os, sys, json, argparse
from datetime import datetime, timedelta

import google.generativeai as genai
import google.api_core.exceptions as gexc

import scheduler_core as core  # reuse the same engine locally
import pytz

# ---------------- Instructions sent to Gemini ----------------
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
- Output JSON ONLY (no backticks, no extra text).
"""

# ---------------- Time helpers ----------------
def _tz(tzname: str):
    try:
        return pytz.timezone(tzname)
    except Exception:
        return pytz.timezone("America/Chicago")

def _coerce_future_start(start_iso: str, tzname: str) -> datetime:
    """Ensure the returned start is in the future; if past, roll forward by weeks until future."""
    tzinfo = _tz(tzname)
    now = datetime.now(tzinfo)
    # robust parse (handles trailing Z)
    dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = tzinfo.localize(dt)
    dt = dt.astimezone(tzinfo)
    while dt <= now:
        dt = dt + timedelta(weeks=1)
    return dt

# ---------------- Response text helpers ----------------
def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if "```" in s:
            s = s.split("```", 1)[0]
    return s.strip()

def _extract_text(resp) -> str:
    """Safely concatenate text from candidates/parts; return '' if none."""
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        out = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                out.append(t)
        if out:
            return "\n".join(out).strip()
    return ""

def _gen_json_or_retry(model, prompt: str) -> str:
    """
    Ask for JSON; if the model returns no parts (blocked/empty), retry without forcing JSON.
    If still empty, raise with finish_reason and prompt_feedback for debugging.
    """
    # First try: force JSON MIME for raw object
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": 512,
            },
        )
    except gexc.ResourceExhausted:
        # Quota: try a cheaper model automatically
        fb_model = genai.GenerativeModel("gemini-2.5-flash")
        resp = fb_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "max_output_tokens": 512},
        )

    txt = _extract_text(resp)
    if not txt:
        # Retry: plain text (some models only return fenced JSON when not forced)
        resp2 = model.generate_content(
            prompt + "\n\nReturn ONLY raw JSON (no backticks). This is a benign scheduling task.",
            generation_config={"max_output_tokens": 512},
        )
        txt = _extract_text(resp2)

        if not txt:
            cand0 = (getattr(resp, "candidates", []) or [None])[0]
            fin = getattr(cand0, "finish_reason", None)
            fb  = getattr(resp, "prompt_feedback", None)
            raise SystemExit(f"Gemini returned no text (finish_reason={fin}, prompt_feedback={fb}).")

    return _strip_code_fence(txt)

# ---------------- Main ----------------
def main():
    if "GOOGLE_API_KEY" not in os.environ:
        print("Set GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Configure SDK
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    ap = argparse.ArgumentParser()
    ap.add_argument("request", help="Natural-language meeting request")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    # Choose model (honor user choice; default to flash)
    model_name = args.model or "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)

    # Ask Gemini for structured JSON
    prompt = INSTRUCTIONS + "\n\nUser request:\n" + args.request
    txt = _gen_json_or_retry(model, prompt)

    # Parse JSON (single attempt; show the raw text if malformed)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        print("Gemini did not return JSON:\n", txt, file=sys.stderr)
        sys.exit(1)

    # Extract fields
    title = (data.get("title") or "Meeting").strip()
    attendees = list({(e or "").strip() for e in data.get("attendees", []) if (e or "").strip()})
    tzname = (data.get("timezone") or os.environ.get("LOCAL_TZ") or "America/Chicago").strip()
    desc = (data.get("description") or "").strip()

    duration = data.get("duration_minutes")
    duration = int(duration) if isinstance(duration, int) and duration > 0 else None

    start_iso = data.get("start")
    end_iso   = data.get("end")
    window    = data.get("window") or {}

    # If user gave a window + duration but no explicit start, pick earliest in window
    if (not start_iso) and window.get("start") and window.get("end") and duration:
        start_iso = window["start"]

    if not start_iso:
        print("No start time parsed (need a time or a window).", file=sys.stderr)
        sys.exit(1)

    # Enforce FUTURE start (fixes wrong-year cases)
    start_dt = _coerce_future_start(start_iso, tzname)

    # Compute end from duration first; else use provided end; else default 30m
    if duration:
        end_dt = start_dt + timedelta(minutes=duration)
    elif end_iso:
        end_dt = _coerce_future_start(end_iso, tzname)
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(minutes=30)
    else:
        end_dt = start_dt + timedelta(minutes=30)

    # Create the Calendar event with a Google Meet link; email invites go out
    evt = core.create_meet_event(
        title=title,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        attendees=attendees,
        description=desc,
        time_zone=tzname,
        send_updates="all",   # ensure invite emails are sent
    )

    print(json.dumps(evt, indent=2))

if __name__ == "__main__":
    main()
