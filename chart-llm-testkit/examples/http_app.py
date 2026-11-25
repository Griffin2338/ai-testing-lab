"""Adapter for a chart app behind an HTTP endpoint.

Config (target is relative to the config file):
    app:
      type: python
      target: examples/http_app.py:HttpChartApp
      kwargs: {url: "http://localhost:8000/chart-code"}

Expected response: {"code": "..."} or {"clarification": "..."}. Change _parse for your API.
The kit runs the returned code in its own sandbox, so the service only has to return code.
"""

from __future__ import annotations

import json
import urllib.request

from chart_testkit.app import Generation


class HttpChartApp:
    def __init__(self, url: str, timeout: float = 120.0):
        self.url, self.timeout = url, timeout
        self.name = f"http app ({url})"

    def generate(self, df, request: str) -> Generation:
        body = json.dumps({"request": request, "csv": df.to_csv(index=False)}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return self._parse(json.loads(r.read()))

    @staticmethod
    def _parse(payload: dict) -> Generation:
        if payload.get("clarification"):
            return Generation(code=None, clarification=payload["clarification"])
        return Generation(code=payload["code"])
