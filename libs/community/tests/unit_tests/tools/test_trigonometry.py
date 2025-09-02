# libs/community/tests/unit_tests/tools/test_trigonometry.py

import json
import math
import pytest
import asyncio

from langchain_community.tools.trigonometry import TrigonometryTool

tool = TrigonometryTool()

def parse(output_str):
    """Helper to parse and return dict from tool output."""
    try:
        return json.loads(output_str)
    except Exception:
        pytest.fail("Output is not valid JSON: %s" % output_str)

def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol

def test_sin_degrees():
    out = parse(tool.run('{"function": "sin", "value": 90, "unit": "degrees"}'))
    assert out["error"] is None
    assert approx(out["result"], 1.0)

def test_cos_radians():
    # cos(pi) = -1
    out = parse(tool.run('{"function": "cos", "value": 3.141592653589793, "unit": "radians"}'))
    assert out["error"] is None
    assert approx(out["result"], -1.0, tol=1e-6)

def test_atan_inverse_degrees():
    # atan(1) => 45 degrees when unit='degrees' for inverse
    out = parse(tool.run('{"function": "atan", "value": 1, "unit": "degrees"}'))
    assert out["error"] is None
    # result should be approx 45.0 (since inverse result was converted to degrees)
    assert approx(out["result"], 45.0, tol=1e-6)
    assert out["unit"] == "degrees"

def test_invalid_function():
    out = parse(tool.run('{"function": "foo", "value": 30, "unit": "degrees"}'))
    assert out["error"] is not None
    assert "supported" in out["error"].lower() or "function" in out["error"].lower()

def test_invalid_json_string():
    out_str = tool.run('not-a-json')
    out = parse(out_str)
    # Because the tool returns error field with message for invalid JSON
    assert out["error"] is not None

def test_domain_error_acos():
    # acos(2) is invalid (domain error), should return error
    out = parse(tool.run('{"function": "acos", "value": 2, "unit": "radians"}'))
    assert out["error"] is not None
    assert "domain" in out["error"].lower() or "math" in out["error"].lower()

def test_dict_input_accepted():
    # ensure dict input also works
    input_dict = {"function": "sin", "value": 30, "unit": "degrees"}
    out = parse(tool.run(input_dict))
    assert out["error"] is None
    assert approx(out["result"], 0.5, tol=1e-6)

def test_async_run():
    async def call():
        return await tool.arun('{"function": "sin", "value": 30, "unit": "degrees"}')
    
    # Use the modern asyncio.run() instead of the deprecated get_event_loop()
    out_str = asyncio.run(call())
    out = parse(out_str)
    assert out["error"] is None
    assert approx(out["result"], 0.5, tol=1e-6)