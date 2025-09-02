# libs/community/tests/unit_tests/tools/test_calendar.py
import json
import pytest
from datetime import datetime, timezone, timedelta
from langchain_community.tools.calendar import CalendarTool

def parse_out(s):
    try:
        return json.loads(s)
    except Exception:
        pytest.fail("Output is not valid JSON: %s" % s)

def test_create_and_get_and_delete_event():
    tool = CalendarTool(storage_path=None)  # in-memory
    # create event with explicit ISO start
    start = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    payload = {
        "action": "create",
        "event": {
            "title": "Unit test meeting",
            "start": start,
            "description": "Testing create/get/delete",
            "timezone": "UTC"
        }
    }
    res = parse_out(tool.run(payload))
    assert res["status"] == "ok"
    assert res["event"] is not None
    eid = res["event"]["id"]

    # get
    g = parse_out(tool.run({"action": "get", "id": eid}))
    assert g["status"] == "ok"
    assert g["event"]["id"] == eid
    assert g["event"]["title"] == "Unit test meeting"

    # delete
    d = parse_out(tool.run({"action": "delete", "id": eid}))
    assert d["status"] == "ok"
    # confirm deletion
    g2 = parse_out(tool.run({"action": "get", "id": eid}))
    assert g2["status"] == "error"

def test_list_and_find_next():
    tool = CalendarTool(storage_path=None)
    now = datetime.now(timezone.utc)
    # event in past
    past_start = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    # event in near future
    future_start = (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    payload1 = {"action": "create", "event": {"title": "Past event", "start": past_start, "timezone": "UTC"}}
    payload2 = {"action": "create", "event": {"title": "Future event", "start": future_start, "timezone": "UTC"}}
    r1 = parse_out(tool.run(payload1))
    r2 = parse_out(tool.run(payload2))
    assert r1["status"] == "ok"
    assert r2["status"] == "ok"

    listing = parse_out(tool.run({"action": "list"}))
    assert listing["status"] == "ok"
    assert listing["count"] >= 2

    nxt = parse_out(tool.run({"action": "find_next"}))
    assert nxt["status"] == "ok"
    # next should be the future event
    assert nxt["event"]["title"] == "Future event"

def test_update_event():
    tool = CalendarTool(storage_path=None)
    start = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    create = parse_out(tool.run({"action": "create", "event": {"title": "To be updated", "start": start}}))
    eid = create["event"]["id"]
    upd_payload = {"action": "update", "id": eid, "event": {"title": "Updated title"}}
    upd = parse_out(tool.run(upd_payload))
    assert upd["status"] == "ok"
    assert upd["event"]["title"] == "Updated title"

def test_find_range():
    tool = CalendarTool(storage_path=None)
    base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    one = (base + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    two = (base + timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    # create two events
    a = parse_out(tool.run({"action": "create", "event": {"title": "A", "start": one}}))
    b = parse_out(tool.run({"action": "create", "event": {"title": "B", "start": two}}))
    start_range = base.isoformat().replace("+00:00", "Z")
    end_range = (base + timedelta(hours=4)).isoformat().replace("+00:00", "Z")
    out = parse_out(tool.run({"action": "find_range", "start": start_range, "end": end_range}))
    assert out["status"] == "ok"
    titles = {e["title"] for e in out["events"]}
    assert "A" in titles and "B" in titles
