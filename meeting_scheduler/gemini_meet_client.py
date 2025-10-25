# gemini_meet_client.py
import os, sys, json, argparse
from datetime import datetime, timedelta

import google.generativeai as genai
import google.api_core.exceptions as gexc
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import scheduler_core as core
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
- Output JSON ONLY (no backticks, no extra text).
"""

# ---------- Safety: allow benign scheduling content ----------
SAFETY_NONE = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUAL_CONTENT:    HarmBlockThreshold.BLOCK_NONE,
}

# ---------- Time helpers ----------
def _tz(tzname: str):
    try:
        return pytz.timezone(tzname)
    except Exception:
        return pytz.timezone("America/Chicago")

def _coerce_future_start(start_iso: str, tzname: str) -> datetime:
    tzinfo = _tz(tzname)
    now = datetime.now(tzinfo)
    dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = tzinfo.localize(dt)
    dt = dt.astimezone(tzinfo)
    while dt <= now:
        dt = dt + timedelta(weeks=1)
    return dt

# ---------- Response helpers ----------
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

def _summarize_debug(resp) -> str:
    try:
        cand0 = (getattr(resp, "candidates", []) or [None])[0]
        fin = getattr(cand0, "finish_reason", None)
        fb  = getattr(resp, "prompt_feedback", None)
        return f"finish_reason={fin}, prompt_feedback={fb}"
    except Exception:
        return "no debug info"

# ---------- Call Gemini robustly ----------
def _try_generate(model, prompt: str, force_json: bool, debug: bool):
    cfg = {"max_output_tokens": 512}
    if force_json:
        cfg["response_mime_type"] = "application/json"

    resp = model.generate_content(
        prompt,
        generation_config=cfg,
        safety_settings=SAFETY_NONE
    )
    txt = _extract_text(resp)
    if debug:
        print("[DEBUG] model:", getattr(model, "model_name", "unknown"),
              "| force_json:", force_json,
              "| extracted_len:", len(txt),
              "|", _summarize_debug(resp),
              file=sys.stderr)
    return txt, resp

def _gen_json_or_retry(models, prompt: str, debug: bool) -> str:
    """
    Iterate over candidate models; for each, try:
      1) forced JSON
      2) non-forced (plain text) with an explicit reminder
    Return stripped JSON text or raise with actionable debug.
    """
    last_dbg = ""
    for m in models:
        # 1) forced JSON
        try:
            txt, resp = _try_generate(m, prompt, force_json=True, debug=debug)
        except gexc.ResourceExhausted:
            # skip to next model if quota hit
            if debug: print("[DEBUG] quota exhausted; trying next model", file=sys.stderr)
            txt, resp = "", None

        if not txt:
            # 2) non-forced
            txt2, resp2 = _try_generate(
                m,
                prompt + "\n\nReturn ONLY raw JSON (no backticks). This is a benign scheduling task.",
                force_json=False,
                debug=debug
            )
            txt = txt2
            resp = resp2 if txt2 else resp

        if txt:
            return _strip_code_fence(txt)

        # collect debug message
        if resp is not None:
            last_dbg = _summarize_debug(resp)

    raise SystemExit(f"Gemini returned no text from all models. Last attempt debug: {last_dbg}")

# ---------- Main ----------
def main():
    if "GOOGLE_API_KEY" not in os.environ:
        print("Set GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    ap = argparse.ArgumentParser()
    ap.add_argument("request", help="Natural-language meeting request")
    ap.add_argument("--model", default=None, help="Gemini model name (e.g., gemini-2.5-flash)")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    # Build model list: prefer user choice, then sensible fallbacks
    primary = args.model or "gemini-2.5-flash"
    candidates = [primary]
    # add fallbacks if not already first
    for alt in ["gemini-flash-latest", "gemini-2.5-pro"]:
        if alt not in candidates:
            candidates.append(alt)

    models = [genai.GenerativeModel(name) for name in candidates]

    prompt = INSTRUCTIONS + "\n\nUser request:\n" + args.request
    txt = _gen_json_or_retry(models, prompt, debug=args.debug)

    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        print("Gemini did not return JSON:\n", txt, file=sys.stderr)
        sys.exit(1)

    title = (data.get("title") or "Meeting").strip()
    attendees = list({(e or "").strip() for e in data.get("attendees", []) if (e or "").strip()})
    tzname = (data.get("timezone") or os.environ.get("LOCAL_TZ") or "America/Chicago").strip()
    desc = (data.get("description") or "").strip()

    duration = data.get("duration_minutes")
    duration = int(duration) if isinstance(duration, int) and duration > 0 else None

    start_iso = data.get("start")
    end_iso   = data.get("end")
    window    = data.get("window") or {}

    if (not start_iso) and window.get("start") and window.get("end") and duration:
        start_iso = window["start"]

    if not start_iso:
        print("No start time parsed (need a time or a window).", file=sys.stderr)
        sys.exit(1)

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
        send_updates="all",  # send email invites
    )

    print(json.dumps(evt, indent=2))

if __name__ == "__main__":
    main()
