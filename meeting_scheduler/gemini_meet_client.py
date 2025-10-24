# gemini_meet_client.py
import os, sys, json, argparse
import google.generativeai as genai
from dateutil import parser as dateparse
import scheduler_core as core  # reuse the same engine locally

INSTRUCTIONS = """You are an assistant that extracts meeting details as JSON.
Given a user request, output ONLY a JSON object with:
{
  "title": "<string, default 'Meeting'>",
  "attendees": ["email1","email2", ...],
  "start": "<ISO or natural-language time in user's locale>",
  "end": "<ISO or natural-language time in user's locale>",
  "timezone": "<IANA tz like America/Chicago>",
  "description": "<notes/agenda, optional>"
}
Assume timezone if missing: America/Chicago. If a range is given, pick the earliest feasible time in the range.
Duration default: 30 minutes if no end given.
Return JSON only, no backticks, no extra text.
"""

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
    attendees = list({e.strip() for e in data.get("attendees", []) if e.strip()})
    tzname = data.get("timezone") or core.DEFAULT_TZ
    start = data.get("start")
    end = data.get("end")

    # Normalize times; if only start given, default 30 minutes
    if not start:
        print("No start time parsed.", file=sys.stderr); sys.exit(1)

    # let scheduler_core handle timezone localization
    if not end:
        # default 30 min
        sd = dateparse.parse(start); ed = sd + core.timedelta(minutes=30) if hasattr(core, "timedelta") else None

    description = data.get("description") or ""

    # Create the meeting
    evt = core.create_meet_event(title=title, start=start, end=end or ed.isoformat(), attendees=attendees, description=description, time_zone=tzname)
    print(json.dumps(evt, indent=2))

if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("Set GOOGLE_API_KEY.", file=sys.stderr); sys.exit(1)
    main()
