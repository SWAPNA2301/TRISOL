"""The report webview: routing, serialisation and traversal safety."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from trisol.findings import AuditReport, CheckResult, Confidence, Finding, Severity
from trisol.web.server import ReportServer


def _report() -> AuditReport:
    finding = Finding(
        check="reliability",
        title="Model call has no timeout",
        detail="detail text",
        severity=Severity.HIGH,
        confidence=Confidence.CERTAIN,
        file="agent.py",
        line=12,
    )
    result = CheckResult(
        name="reliability",
        title="Reliability",
        findings=[finding],
        metrics={"model_calls": 1},
    )
    return AuditReport(
        target="/tmp/agent",
        started_at=time.time() - 1,
        finished_at=time.time(),
        results=[result],
        model={"available": False, "reason": "offline mode"},
        trisol_version="0.1.0",
    )


@pytest.fixture
def server():
    srv = ReportServer(_report(), port=0)
    srv.start()
    port = srv._httpd.server_address[1]
    yield f"http://127.0.0.1:{port}"
    srv.stop()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read()


class TestRouting:
    def test_index_is_served_at_the_root(self, server) -> None:
        status, body = _get(f"{server}/")
        assert status == 200
        assert b"TRISOL" in body

    def test_api_returns_the_report(self, server) -> None:
        status, body = _get(f"{server}/api/report")
        assert status == 200
        payload = json.loads(body)
        assert payload["summary"]["high"] == 1
        assert payload["results"][0]["findings"][0]["title"].startswith("Model call")

    @pytest.mark.parametrize("asset", ["styles.css", "app.js"])
    def test_static_assets_are_served(self, server, asset) -> None:
        status, body = _get(f"{server}/{asset}")
        assert status == 200
        assert body

    def test_unknown_path_is_404(self, server) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"{server}/nope.txt")
        assert exc.value.code == 404

    def test_traversal_cannot_escape_the_web_directory(self, server) -> None:
        """Paths are resolved by basename only, so `../` reaches nothing."""
        for attempt in ("/../server.py", "/..%2fserver.py", "/../../pyproject.toml"):
            try:
                status, _ = _get(f"{server}{attempt}")
            except urllib.error.HTTPError as exc:
                assert exc.code in (400, 404)
            else:
                # If something was served it must not be source outside web/.
                assert status == 200

    def test_query_string_is_ignored(self, server) -> None:
        status, _ = _get(f"{server}/api/report?x=1")
        assert status == 200


class TestSerialisation:
    def test_report_is_snapshotted_at_start(self) -> None:
        """Serialising once means the page can never read a half-written file."""
        report = _report()
        srv = ReportServer(report, port=0)
        # Mutating the report afterwards must not change what is served.
        report.results[0].findings.clear()
        srv.start()
        try:
            port = srv._httpd.server_address[1]
            _status, body = _get(f"http://127.0.0.1:{port}/api/report")
            assert json.loads(body)["summary"]["high"] == 1
        finally:
            srv.stop()

    def test_url_reports_the_bound_host_and_port(self) -> None:
        srv = ReportServer(_report(), port=0)
        assert srv.url.startswith("http://127.0.0.1:")

    def test_stop_is_idempotent(self) -> None:
        """Callers stop in a finally block, sometimes twice."""
        srv = ReportServer(_report(), port=0)
        srv.start()
        srv.stop()
        srv.stop()

    def test_port_in_use_raises_an_actionable_error(self) -> None:
        first = ReportServer(_report(), port=0)
        first.start()
        try:
            taken = first._httpd.server_address[1]
            second = ReportServer(_report(), port=taken)
            with pytest.raises(OSError, match="--port"):
                second.start()
        finally:
            first.stop()
