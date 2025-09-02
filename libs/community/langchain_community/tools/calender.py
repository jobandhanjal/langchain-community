# libs/community/langchain_community/tools/calendar.py
from __future__ import annotations
from typing import Any, Dict, Optional, List, Union
from langchain.tools import BaseTool
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
import sqlite3
from zoneinfo import ZoneInfo  # stdlib (py3.9+); fallback to pytz not implemented here
from dateutil import parser as dateutil_parser  # widely available; used as a fallback

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _isoformat(dt: datetime) -> str:
    """Return ISO 8601 UTC string for a timezone-aware datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat().replace("+00:00", "Z")


def _ensure_tz(dt: datetime, tz: Optional[str]) -> datetime:
    """Attach timezone if naive; return aware datetime."""
    if dt.tzinfo is None:
        if tz:
            try:
                dt = dt.replace(tzinfo=ZoneInfo(tz))
            except Exception:
                # if timezone not recognized, default to UTC
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt


class CalendarTool(BaseTool):
    """
    CalendarTool - Minimal calendar CRUD and query tool for agent workflows.

    Features (core):
      - Create, List, Get, Update, Delete events
      - Find next event and list events in a time-range
      - Accepts JSON string or dict inputs
      - Returns structured JSON output (status, event(s), error)
      - Optional persistent store using SQLite

    Input schema (JSON string or dict):
    {
      "action": "create|list|get|delete|update|find_next|find_range",
      # for create/update:
      "event": {
          "title": "Meeting",
          "start": "2025-09-03T15:00:00Z" | "tomorrow 5pm",
          "end": "2025-09-03T16:00:00Z",   # optional
          "description": "...",
          "timezone": "Asia/Kolkata",      # optional, defaults to tool.default_timezone
          "metadata": { ... }              # optional
      },
      # for get/delete/update:
      "id": "<event-id>",
      # for find_range:
      "start": "2025-09-03T00:00:00Z",
      "end": "2025-09-04T00:00:00Z",
      # optional: "limit": int
    }

    Output (JSON string):
    {
      "status": "ok" | "error",
      "error": null | "error message",
      "event": { ... } | null,
      "events": [ ... ] | null,
      "count": int | null
    }
    """

    name = "calendar_tool"
    description = (
        "Simple CalendarTool: create/list/get/update/delete events and query next or range. "
        "Accepts JSON string or dict (see docstring for schema). Optionally persists events into a SQLite file when 'storage_path' set."
    )

    def __init__(self, storage_path: Optional[str] = None, default_timezone: str = "UTC"):
        """
        storage_path: optional path to sqlite db file; if None, keep events in-memory.
        default_timezone: default timezone string (IANA) applied to naive datetimes.
        """
        self.default_timezone = default_timezone
        self._use_sqlite = storage_path is not None
        self._conn: Optional[sqlite3.Connection] = None
        self._in_memory: Dict[str, Dict[str, Any]] = {}

        if self._use_sqlite:
            self._conn = sqlite3.connect(storage_path, check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    start TEXT,
                    end TEXT,
                    timezone TEXT,
                    metadata TEXT
                )
                """
            )
            self._conn.commit()

    # ---------- Persistence helpers ----------
    def _persist_insert(self, evt: Dict[str, Any]) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO events (id,title,description,start,end,timezone,metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                evt["id"],
                evt.get("title"),
                evt.get("description"),
                evt.get("start"),
                evt.get("end"),
                evt.get("timezone"),
                json.dumps(evt.get("metadata", {})),
            ),
        )
        self._conn.commit()

    def _persist_update(self, evt: Dict[str, Any]) -> None:
        assert self._conn is not None
        self._conn.execute(
            "UPDATE events SET title=?, description=?, start=?, end=?, timezone=?, metadata=? WHERE id=?",
            (
                evt.get("title"),
                evt.get("description"),
                evt.get("start"),
                evt.get("end"),
                evt.get("timezone"),
                json.dumps(evt.get("metadata", {})),
                evt["id"],
            ),
        )
        self._conn.commit()

    def _persist_delete(self, event_id: str) -> None:
        assert self._conn is not None
        self._conn.execute("DELETE FROM events WHERE id=?", (event_id,))
        self._conn.commit()

    def _persist_list_all(self) -> List[Dict[str, Any]]:
        assert self._conn is not None
        cur = self._conn.execute("SELECT id,title,description,start,end,timezone,metadata FROM events")
        rows = cur.fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "title": r[1],
                    "description": r[2],
                    "start": r[3],
                    "end": r[4],
                    "timezone": r[5],
                    "metadata": json.loads(r[6]) if r[6] else {},
                }
            )
        return out

    def _persist_get(self, event_id: str) -> Optional[Dict[str, Any]]:
        assert self._conn is not None
        cur = self._conn.execute("SELECT id,title,description,start,end,timezone,metadata FROM events WHERE id=?", (event_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "start": row[3],
            "end": row[4],
            "timezone": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }

    # ---------- Parsing and normalization ----------
    def _parse_datetime(self, value: Union[str, datetime], tz: Optional[str]) -> datetime:
        """
        Parse an incoming date/time (ISO string or human string or datetime) into a timezone-aware UTC datetime.
        Uses dateutil.parser.parse as fallback. If timezone info is missing, attach tz (default_timezone or provided tz).
        Returns datetime in UTC.
        """
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                # Prefer dateutil (robust for ISO and many formats)
                dt = dateutil_parser.parse(value)
            except Exception as e:
                raise ValueError(f"Unable to parse datetime string: {value}; error: {e}")
        else:
            raise ValueError("start/end must be a datetime or string")

        # Attach timezone if naive
        dt = _ensure_tz(dt, tz or self.default_timezone)

        # Convert to UTC for storage & comparison
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc

    def _normalize_input(self, q: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Return normalized dict from JSON string or dict; perform basic validation."""
        if isinstance(q, str):
            try:
                parsed = json.loads(q)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input: {e}")
        elif isinstance(q, dict):
            parsed = dict(q)
        else:
            raise ValueError("Input must be JSON string or dict.")

        action = parsed.get("action")
        if not action or not isinstance(action, str):
            raise ValueError("Missing or invalid 'action' field.")

        out: Dict[str, Any] = {"action": action.lower()}

        # forward common fields
        if "id" in parsed:
            out["id"] = parsed["id"]

        if "limit" in parsed:
            out["limit"] = int(parsed["limit"])

        # event content
        if "event" in parsed and isinstance(parsed["event"], dict):
            out["event"] = dict(parsed["event"])

        # range queries
        if "start" in parsed:
            out["start"] = parsed["start"]
        if "end" in parsed:
            out["end"] = parsed["end"]

        return out

    # ---------- Event operations (in-memory) ----------
    def _create_event_in_memory(self, evt: Dict[str, Any]) -> Dict[str, Any]:
        event_id = str(uuid.uuid4())
        tz = evt.get("timezone") or self.default_timezone
        start_dt = self._parse_datetime(evt["start"], tz)
        end_dt = None
        if "end" in evt and evt["end"] is not None:
            end_dt = self._parse_datetime(evt["end"], tz)

        stored = {
            "id": event_id,
            "title": evt.get("title"),
            "description": evt.get("description"),
            "start": _isoformat(start_dt),
            "end": _isoformat(end_dt) if end_dt else None,
            "timezone": tz,
            "metadata": evt.get("metadata", {}),
        }

        if self._use_sqlite:
            self._persist_insert(stored)
        else:
            self._in_memory[event_id] = stored
        return stored

    def _list_events_in_memory(self) -> List[Dict[str, Any]]:
        if self._use_sqlite:
            return self._persist_list_all()
        return list(self._in_memory.values())

    def _get_event_in_memory(self, event_id: str) -> Optional[Dict[str, Any]]:
        if self._use_sqlite:
            return self._persist_get(event_id)
        return self._in_memory.get(event_id)

    def _delete_event_in_memory(self, event_id: str) -> bool:
        if self._use_sqlite:
            if self._persist_get(event_id) is None:
                return False
            self._persist_delete(event_id)
            return True
        return self._in_memory.pop(event_id, None) is not None

    def _update_event_in_memory(self, event_id: str, changes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        existing = self._get_event_in_memory(event_id)
        if not existing:
            return None
        # parse/merge changes
        if "start" in changes:
            tz = changes.get("timezone", existing.get("timezone", self.default_timezone))
            try:
                start_dt = self._parse_datetime(changes["start"], tz)
                existing["start"] = _isoformat(start_dt)
            except Exception as e:
                raise ValueError(f"Invalid start value: {e}")
        if "end" in changes:
            tz = changes.get("timezone", existing.get("timezone", self.default_timezone))
            if changes["end"] is None:
                existing["end"] = None
            else:
                existing["end"] = _isoformat(self._parse_datetime(changes["end"], tz))
        if "title" in changes:
            existing["title"] = changes["title"]
        if "description" in changes:
            existing["description"] = changes["description"]
        if "timezone" in changes:
            existing["timezone"] = changes["timezone"]
        if "metadata" in changes:
            existing["metadata"] = changes["metadata"]

        if self._use_sqlite:
            self._persist_update(existing)
        else:
            self._in_memory[event_id] = existing
        return existing

    # ---------- Query helpers ----------
    def _events_in_range(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        events = self._list_events_in_memory()
        out = []
        for e in events:
            e_start = dateutil_parser.isoparse(e["start"])
            e_end = dateutil_parser.isoparse(e["end"]) if e.get("end") else None
            if (e_start >= start and e_start <= end) or (e_end and e_end >= start and e_end <= end):
                out.append(e)
        # sort by start ascending
        out.sort(key=lambda x: dateutil_parser.isoparse(x["start"]))
        return out

    def _find_next_event(self, now_dt: datetime) -> Optional[Dict[str, Any]]:
        events = self._list_events_in_memory()
        upcoming = []
        for e in events:
            e_start = dateutil_parser.isoparse(e["start"])
            if e_start >= now_dt:
                upcoming.append((e_start, e))
        if not upcoming:
            return None
        upcoming.sort(key=lambda x: x[0])
        return upcoming[0][1]

    # ---------- Execution ----------

    def _format_response(self, status: str = "ok", error: Optional[str] = None, event: Optional[Dict[str, Any]] = None, events: Optional[List[Dict[str, Any]]] = None) -> str:
        payload = {
            "status": status,
            "error": error,
            "event": event,
            "events": events,
            "count": len(events) if events is not None else (1 if event is not None else 0),
        }
        return json.dumps(payload)

    def _run(self, query: Union[str, Dict[str, Any]]) -> str:
        """
        Main synchronous entrypoint invoked by LangChain agents. Accepts dict or JSON string.
        """
        try:
            q = self._normalize_input(query)
            action = q["action"]

            # CREATE
            if action == "create":
                evt = q.get("event")
                if not evt:
                    return self._format_response("error", "Missing 'event' payload for create.")
                stored = self._create_event_in_memory(evt)
                return self._format_response("ok", None, event=stored)

            # LIST
            if action == "list":
                items = self._list_events_in_memory()
                return self._format_response("ok", None, events=items)

            # GET
            if action == "get":
                eid = q.get("id")
                if not eid:
                    return self._format_response("error", "Missing 'id' for get.")
                item = self._get_event_in_memory(eid)
                if not item:
                    return self._format_response("error", f"No event with id {eid}.")
                return self._format_response("ok", None, event=item)

            # DELETE
            if action == "delete":
                eid = q.get("id")
                if not eid:
                    return self._format_response("error", "Missing 'id' for delete.")
                ok = self._delete_event_in_memory(eid)
                if not ok:
                    return self._format_response("error", f"No event with id {eid}.")
                return self._format_response("ok", None, event={"id": eid})

            # UPDATE
            if action == "update":
                eid = q.get("id")
                changes = q.get("event")
                if not eid or not changes:
                    return self._format_response("error", "Missing 'id' or 'event' for update.")
                updated = self._update_event_in_memory(eid, changes)
                if not updated:
                    return self._format_response("error", f"No event with id {eid}.")
                return self._format_response("ok", None, event=updated)

            # FIND_RANGE
            if action == "find_range":
                if "start" not in q or "end" not in q:
                    return self._format_response("error", "Missing 'start' or 'end' for find_range.")
                start_dt = self._parse_datetime(q["start"], None)
                end_dt = self._parse_datetime(q["end"], None)
                items = self._events_in_range(start_dt, end_dt)
                return self._format_response("ok", None, events=items)

            # FIND_NEXT
            if action == "find_next":
                now_dt = datetime.now(timezone.utc)
                nxt = self._find_next_event(now_dt)
                if not nxt:
                    return self._format_response("ok", None, events=[])
                return self._format_response("ok", None, event=nxt)

            return self._format_response("error", f"Unsupported action '{action}'.")

        except ValueError as e:
            logger.debug("Validation error: %s", e)
            return self._format_response("error", str(e))
        except Exception as e:
            logger.exception("Unexpected error in CalendarTool: %s", e)
            return self._format_response("error", f"Unexpected error: {e}")

    async def _arun(self, query: Union[str, Dict[str, Any]]) -> str:
        import asyncio

        return await asyncio.to_thread(self._run, query)
