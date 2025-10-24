# scheduler_core.py
from __future__ import annotations
import os, json, pathlib, typing as T
from datetime import datetime, timedelta
from dateutil import parser as dateparse, tz
import pytz

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]
DATA_DIR = pathlib.Path(os.environ.get("MEET_SCHED_DATA_DIR", "~/.mcp/meet_sched")).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_PATH = DATA_DIR / "token.json"
CLIENT_SECRET_PATH = pathlib.Path(os.environ.get("GOOGLE_CLIENT_SECRET_JSON", "credentials/client_secret.json"))

DEFAULT_CAL_ID = "primary"
DEFAULT_TZ = os.environ.get("LOCAL_TZ", "America/Chicago")

def _rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = pytz.timezone(DEFAULT_TZ).localize(dt)
    return dt.astimezone(pytz.UTC).isoformat()

def _get_creds() -> Credentials:
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
            # Colab/CLI friendly console flow:
            creds = flow.run_console()
        TOKEN_PATH.write_text(creds.to_json())
    return creds

def _svc():
    return build("calendar", "v3", credentials=_get_creds())

# ---------- Public helpers ----------

def list_calendars() -> list[dict]:
    service = _svc()
    items = service.calendarList().list().execute().get("items", [])
    return [{"id": i["id"], "summary": i.get("summary"), "timeZone": i.get("timeZone")} for i in items]

def find_free_slots(
    attendees: list[str],
    start: str,
    end: str,
    duration_minutes: int = 30,
    time_zone: str = DEFAULT_TZ
) -> dict:
    """Return a simple availability grid using freeBusy for the primary calendar + attendees."""
    service = _svc()
    # parse window
    tzinfo = pytz.timezone(time_zone)
    start_dt = dateparse.parse(start)
    end_dt = dateparse.parse(end)
    if start_dt.tzinfo is None: start_dt = tzinfo.localize(start_dt)
    if end_dt.tzinfo is None: end_dt = tzinfo.localize(end_dt)

    fb_req = {
        "timeMin": start_dt.astimezone(pytz.UTC).isoformat(),
        "timeMax": end_dt.astimezone(pytz.UTC).isoformat(),
        "items": [{"id": DEFAULT_CAL_ID}] + [{"id": e} for e in attendees]
    }
    fb = service.freebusy().query(body=fb_req).execute()["calendars"]

    # Build busy intervals per attendee
    busy_map = {k: [(dateparse.parse(b["start"]), dateparse.parse(b["end"])) for b in v.get("busy", [])]
                for k, v in fb.items()}

    # Candidate slots
    step = timedelta(minutes=duration_minutes)
    slots = []
    cur = start_dt
    while cur + step <= end_dt:
        slot = (cur, cur + step)
        slots.append(slot)
        cur += step  # step slots back-to-back; change to 15min if you want overlapping checks

    def overlaps(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    available = []
    for s in slots:
        ok = True
        for who, intervals in busy_map.items():
            if any(overlaps(s, (i[0], i[1])) for i in intervals):
                ok = False; break
        if ok:
            available.append({"start": s[0].isoformat(), "end": s[1].isoformat()})

    return {
        "window": {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "tz": time_zone},
        "duration_minutes": duration_minutes,
        "attendees": attendees,
        "available": available[:50]
    }

def create_meet_event(
    title: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
    description: str | None = None,
    calendar_id: str = DEFAULT_CAL_ID,
    time_zone: str = DEFAULT_TZ,
    send_updates: str = "all"
) -> dict:
    """
    Create a Calendar event with an auto-generated Google Meet link.
    Returns event id, htmlLink, meet link, etc.
    """
    attendees = attendees or []
    tzinfo = pytz.timezone(time_zone)
    start_dt = dateparse.parse(start)
    end_dt = dateparse.parse(end)
    if start_dt.tzinfo is None: start_dt = tzinfo.localize(start_dt)
    if end_dt.tzinfo is None: end_dt = tzinfo.localize(end_dt)

    body = {
        "summary": title,
        "description": description or "",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": time_zone},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": time_zone},
        "attendees": [{"email": e} for e in attendees],
        "conferenceData": {
            "createRequest": {"requestId": f"meet-{int(datetime.now().timestamp())}"}
        }
    }
    service = _svc()
    event = service.events().insert(
        calendarId=calendar_id,
        body=body,
        conferenceDataVersion=1,
        sendUpdates=send_updates
    ).execute()

    meet_link = None
    conf = event.get("conferenceData", {})
    if conf.get("entryPoints"):
        for ep in conf["entryPoints"]:
            if ep.get("entryPointType") == "video":
                meet_link = ep.get("uri")

    return {
        "id": event["id"],
        "htmlLink": event.get("htmlLink"),
        "hangoutLink": event.get("hangoutLink"),  # legacy field; often also set
        "meetLink": meet_link,
        "start": event["start"],
        "end": event["end"],
        "attendees": [a.get("email") for a in event.get("attendees", [])],
        "calendarId": calendar_id
    }

def list_events(calendar_id: str = DEFAULT_CAL_ID, time_min: str | None = None, time_max: str | None = None, max_results: int = 20) -> list[dict]:
    service = _svc()
    args = {"calendarId": calendar_id, "singleEvents": True, "orderBy": "startTime", "maxResults": max_results}
    if time_min: args["timeMin"] = dateparse.parse(time_min).astimezone(pytz.UTC).isoformat()
    if time_max: args["timeMax"] = dateparse.parse(time_max).astimezone(pytz.UTC).isoformat()
    items = service.events().list(**args).execute().get("items", [])
    out = []
    for e in items:
        meet = None
        conf = e.get("conferenceData", {})
        if conf.get("entryPoints"):
            for ep in conf["entryPoints"]:
                if ep.get("entryPointType") == "video":
                    meet = ep.get("uri")
                    break
        out.append({
            "id": e["id"],
            "summary": e.get("summary"),
            "start": e.get("start"),
            "end": e.get("end"),
            "meetLink": meet or e.get("hangoutLink"),
            "htmlLink": e.get("htmlLink")
        })
    return out
