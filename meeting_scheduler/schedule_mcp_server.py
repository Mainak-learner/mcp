# schedule_mcp_server.py
from __future__ import annotations
import json
from mcp.server.fastmcp import FastMCP
import scheduler_core as core

mcp = FastMCP("meet-scheduler")

@mcp.tool()
def list_calendars() -> str:
    "List the user's calendars."
    return json.dumps(core.list_calendars(), ensure_ascii=False)

@mcp.tool()
def find_free(attendees: list[str], start_window: str, end_window: str, duration_minutes: int = 30, time_zone: str | None = None) -> str:
    "Find free slots for all attendees in the window. Times are ISO8601 or natural language (e.g. '2025-10-28 10:00')."
    out = core.find_free_slots(attendees, start_window, end_window, duration_minutes=duration_minutes, time_zone=time_zone or core.DEFAULT_TZ)
    return json.dumps(out, ensure_ascii=False)

@mcp.tool()
def create_meeting(title: str, start: str, end: str, attendees: list[str], description: str = "", time_zone: str | None = None, send_updates: str = "all") -> str:
    "Create a Google Calendar event with a Meet link and invite attendees."
    evt = core.create_meet_event(
        title=title,
        start=start,
        end=end,
        attendees=attendees,
        description=description,
        time_zone=time_zone or core.DEFAULT_TZ,
        send_updates=send_updates
    )
    return json.dumps(evt, ensure_ascii=False)

@mcp.tool()
def list_events(time_min: str | None = None, time_max: str | None = None, max_results: int = 10) -> str:
    "List upcoming events (optionally within a window)."
    return json.dumps(core.list_events(time_min=time_min, time_max=time_max, max_results=max_results), ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
