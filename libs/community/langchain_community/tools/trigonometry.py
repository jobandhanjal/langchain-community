# libs/community/langchain_community/tools/trigonometry.py

from __future__ import annotations
from typing import Union, Dict, Any
from langchain.tools import BaseTool
import math
import json
import asyncio
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TrigonometryTool(BaseTool):
    """
    TrigonometryTool: Evaluate basic trigonometric functions with explicit
    support for radians and degrees. Designed to produce structured JSON outputs
    that are easy for agents to parse.

    Supported functions:
      - forward: sin, cos, tan
      - inverse: asin, acos, atan

    Input:
      - either a JSON string or a python dict with keys:
        {
          "function": "sin|cos|tan|asin|acos|atan",
          "value": number,
          "unit": "radians" | "degrees"    # optional, default "radians"
        }

    Output (JSON string):
      {
        "function": "<name>",
        "input": { ...original input... },
        "result": <number|null>,
        "unit": "radians" | "degrees" | null,  # unit of 'result' (degrees for inverse when requested)
        "raw_result": <raw_numeric_result_from_math_module|null>,
        "error": <null | "error message">
      }
    """

    name = "trigonometry_tool"
    description = (
        "Evaluate basic trigonometric functions (sin, cos, tan, asin, acos, atan) "
        "with explicit support for 'radians' and 'degrees'. "
        "Accepts either a JSON string or a dict with keys: "
        '{"function": "sin|cos|tan|asin|acos|atan", "value": number, "unit": "radians|degrees"}'
    )

    # mapping of function name -> math module callable
    _FUNC_MAP = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
    }

    def _normalize_input(self, query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse and validate input; return normalized dict."""
        if isinstance(query, str):
            try:
                parsed = json.loads(query)
            except json.JSONDecodeError as e:
                logger.debug("JSON decode error for query: %s; error: %s", query, e)
                raise ValueError(f"Invalid JSON input: {str(e)}")
        elif isinstance(query, dict):
            parsed = dict(query)  # shallow copy
        else:
            raise ValueError("Input must be a JSON string or a dict.")

        # required fields
        func = parsed.get("function")
        if not func or not isinstance(func, str):
            raise ValueError("Missing or invalid 'function' (must be a string).")

        val = parsed.get("value")
        if val is None or not isinstance(val, (int, float)):
            raise ValueError("Missing or invalid 'value' (must be a number).")

        unit = parsed.get("unit", "radians")
        if not isinstance(unit, str):
            raise ValueError("'unit' must be a string 'radians' or 'degrees' if provided.")
        unit = unit.lower()
        if unit not in ("radians", "degrees"):
            raise ValueError("unit must be 'radians' or 'degrees'.")

        normalized = {"function": func.lower(), "value": float(val), "unit": unit}
        logger.debug("Normalized input: %s", normalized)
        return normalized

    def _compute(self, func: str, value: float, unit: str) -> Dict[str, Any]:
        """
        Compute the trig result.
        - For forward functions (sin, cos, tan), convert degrees -> radians if needed.
        - For inverse functions (asin, acos, atan), compute in radians then convert to degrees if requested.
        Returns a dict with result info.
        """
        if func not in self._FUNC_MAP:
            raise ValueError(f"Function '{func}' not supported. Supported: {list(self._FUNC_MAP.keys())}")

        fn = self._FUNC_MAP[func]
        is_inverse = func in ("asin", "acos", "atan")

        # convert input to radians for math functions if necessary (only for forward functions)
        input_for_math = value
        if not is_inverse and unit == "degrees":
            input_for_math = math.radians(value)
            logger.debug("Converted input degrees->radians: %s -> %s", value, input_for_math)

        # perform computation; capture ValueError from domain issues
        try:
            raw_result = fn(input_for_math)
            logger.debug("Raw result from math.%s(%s) = %s", func, input_for_math, raw_result)
        except ValueError as e:
            logger.warning("Math domain error for %s(%s): %s", func, input_for_math, e)
            raise

        # For inverse functions, math returns radians; optionally convert to degrees
        result_unit = None
        final_result = raw_result
        if is_inverse:
            # result is in radians by default; if caller asked 'degrees' treat result as degrees
            if unit == "degrees":
                final_result = math.degrees(raw_result)
                result_unit = "degrees"
                logger.debug("Converted inverse result radians->degrees: %s -> %s", raw_result, final_result)
            else:
                result_unit = "radians"
        else:
            # forward trig results are unitless ratios
            result_unit = None

        return {
            "result": final_result,
            "raw_result": raw_result,
            "unit": result_unit,
        }

    def _format_response(self, func: str, original_input: Dict[str, Any], compute_out: Dict[str, Any], error: Union[str, None] = None) -> str:
        """Return the standardized JSON string response."""
        response = {
            "function": func,
            "input": original_input,
            "result": None if error else compute_out.get("result"),
            "unit": None if error else compute_out.get("unit"),
            "raw_result": None if error else compute_out.get("raw_result"),
            "error": error,
        }
        return json.dumps(response)

    def _run(self, query: Union[str, Dict[str, Any]]) -> str:
        """Synchronous execution entrypoint used by agents.

        Args:
            query: Either a JSON string or a dict with the following keys:
                function: One of 'sin', 'cos', 'tan', 'asin', 'acos', or 'atan'.
                value: A number to compute the trigonometric function for.
                unit: Either 'radians' or 'degrees'. Optional, defaults to 'radians'.

        Returns:
            A JSON string containing the result of the computation,
            including the original input and any errors.
        """
        normalized = None
        try:
            # Step 1: Normalize the input. This can raise a ValueError.
            normalized = self._normalize_input(query)
            func = normalized["function"]
            value = normalized["value"]
            unit = normalized["unit"]

            # Step 2: Compute the result. This can also raise a ValueError (e.g., domain error).
            compute_out = self._compute(func, value, unit)
            return self._format_response(func, normalized, compute_out, error=None)

        except ValueError as e:
            # If normalization succeeded, 'normalized' will be a dict.
            if normalized:
                func = normalized.get("function", "unknown")
                input_val = normalized
            # If normalization failed, 'query' holds the original malformed input.
            else:
                func = "unknown"
                # For consistency, we still create a dict for the 'input' field.
                input_val = {"raw_query": query} if isinstance(query, str) else query

            return self._format_response(func, input_val, {}, error=str(e))
        except Exception as e:
            # Unexpected errors
            logger.exception("Unexpected error in TrigonometryTool: %s", e)
            input_val = {"raw_query": query} if isinstance(query, str) else query
            return self._format_response("unknown", input_val, {}, error=f"Unexpected error: {str(e)}")

    async def _arun(self, query: Union[str, Dict[str, Any]]) -> str:
        """Asynchronous execution entrypoint used by agents.

        Args:
            query: Either a JSON string or a dict with the following keys:
                function: One of 'sin', 'cos', 'tan', 'asin', 'acos', or 'atan'.
                value: A number to compute the trigonometric function for.
                unit: Either 'radians' or 'degrees'. Optional, defaults to 'radians'.

        Returns:
            A JSON string containing the result of the computation,
            including the original input and any errors.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)