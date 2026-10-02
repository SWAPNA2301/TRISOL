"""The live dashboard: a local web UI that watches an audit happen.

The server starts *before* the audit, so the link is printed immediately and
the page shows each check move from pending to running to done as it happens.
State is pushed to the browser with server-sent events; nothing polls.

From the page a user can re-run the audit and apply the automatic fixes. Applying
fixes re-runs the audit straight away, so the score history shows the effect --
audit, fix, verify -- without leaving the browser.

Standard library only. Writes are accepted only from the page's own origin, and
the only writes possible are "run the audit" and "apply these exact patches",
both of which the CLI can already do.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..findings import AuditReport, CheckResult

__all__ = ["LiveDashboard", "ReportServer", "serve_live", "serve_report"]

_WEB_DIR = Path(__file__).parent
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}
_MAX_LOG = 300
_MAX_BODY = 64_000


class LiveDashboard:
    """Owns one target's audit state and serves it to the browser."""

    def __init__(
        self,
        target: Path | str,
        *,
        host: str = "127.0.0.1",
        port: int = 8899,
        offline: bool = False,
        claude_bin: str | None = None,
        initial_report: AuditReport | None = None,
    ):
        self.target = str(Path(target).resolve())
        self.host = host
        self.port = port
        self.offline = offline
        self.claude_bin = claude_bin
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._stopping = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._report: AuditReport | None = None
        self._run_thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "version": 0,
            "status": "idle",
            "target": self.target,
            "run": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "offline": offline,
            "checks": [],
            "log": [],
            "report": None,
            "fixes": [],
            "last_fix": None,
            "history": [],
        }
        if initial_report is not None:
            # A finished report handed in by `trisol audit --serve` or the menu:
            # show it as a completed run, still re-runnable from the page.
            self._finish(initial_report, run=1)

    # -- state ---------------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state, default=str))

    def _update(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)
            self._state["version"] += 1
            self._changed.notify_all()

    def _log(self, text: str, kind: str = "info") -> None:
        with self._lock:
            self._state["log"].append({"t": time.time(), "text": text, "kind": kind})
            del self._state["log"][:-_MAX_LOG]
            self._state["version"] += 1
            self._changed.notify_all()

    def _set_check(self, name: str, **changes: Any) -> None:
        with self._lock:
            for check in self._state["checks"]:
                if check["name"] == name:
                    check.update(changes)
            self._state["version"] += 1
            self._changed.notify_all()

    def wait_for_change(self, version: int, timeout: float) -> int:
        """Block until the state version moves past ``version`` or timeout."""
        with self._lock:
            self._changed.wait_for(
                lambda: self._state["version"] != version or self._stopping.is_set(),
                timeout=timeout,
            )
            return int(self._state["version"])

    # -- running ---------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._state["status"] == "running"

    def start_run(self) -> bool:
        """Start an audit in the background. False if one is already running."""
        with self._lock:
            if self._state["status"] == "running":
                return False
            self._state["status"] = "running"
        self._run_thread = threading.Thread(target=self._run, daemon=True)
        self._run_thread.start()
        return True

    def _run(self) -> None:
        from ..audit import audit  # noqa: PLC0415 - avoids a cycle at import time
        from ..registry import ALL_CHECKS  # noqa: PLC0415

        run = int(self._state["run"]) + 1
        self._update(
            run=run,
            started_at=time.time(),
            finished_at=None,
            error=None,
            checks=[
                {"name": c.name, "title": c.title, "state": "pending", "findings": 0,
                 "duration_ms": 0, "skip_reason": ""}
                for c in ALL_CHECKS
            ],
        )
        self._log(f"run {run}: auditing {self.target}", "start")

        def on_progress(stage: str, name: str) -> None:
            if stage == "discover":
                self._log("discovering agent code, prompts, model calls and data")
            else:
                self._set_check(name, state="running")
                self._log(f"running {name}")

        def on_result(result: CheckResult) -> None:
            state = "skipped" if result.skipped else "done"
            self._set_check(
                result.name, state=state, findings=len(result.findings),
                duration_ms=result.duration_ms, skip_reason=result.skip_reason,
            )
            if result.skipped:
                self._log(f"{result.name} skipped: {result.skip_reason}", "skip")
            else:
                noun = "finding" if len(result.findings) == 1 else "findings"
                self._log(f"{result.name} done: {len(result.findings)} {noun} "
                          f"in {result.duration_ms}ms", "done")

        try:
            report = audit(
                self.target,
                offline=self.offline,
                claude_bin=self.claude_bin,
                on_progress=on_progress,
                on_result=on_result,
            )
        except Exception as exc:  # surfaced on the page; the server stays up
            self._update(status="error", error=f"{exc.__class__.__name__}: {exc}",
                         finished_at=time.time())
            self._log(f"run {run} failed: {exc}", "error")
            return
        self._finish(report, run=run)

    def _finish(self, report: AuditReport, *, run: int) -> None:
        self._report = report
        payload = report.as_dict()
        fixes = [
            {
                "id": index,
                "title": f.title,
                "severity": f.severity.value,
                "file": f.patch.file if f.patch else f.file,
                "line": f.line,
                "old": f.patch.old if f.patch else "",
                "new": f.patch.new if f.patch else "",
                "explanation": f.patch.explanation if f.patch else "",
            }
            for index, f in enumerate(report.fixable)
        ]
        with self._lock:
            history = list(self._state["history"])
            history.append(
                {
                    "run": run,
                    "overall": payload["scorecard"]["overall"],
                    "total": payload["summary"]["total"],
                    "finished_at": report.finished_at,
                }
            )
            if not self._state["checks"]:
                self._state["checks"] = [
                    {"name": r.name, "title": r.title,
                     "state": "skipped" if r.skipped else "done",
                     "findings": len(r.findings), "duration_ms": r.duration_ms,
                     "skip_reason": r.skip_reason}
                    for r in report.results
                ]
            self._state.update(
                status="done", run=run, report=payload, fixes=fixes,
                history=history[-30:], finished_at=report.finished_at,
            )
            self._state["version"] += 1
            self._changed.notify_all()
        overall = payload["scorecard"]["overall"]
        score = f", score {overall}/100" if overall is not None else ""
        self._log(f"run {run} complete: {payload['summary']['total']} findings{score}", "finish")

    def apply_fixes(self, ids: list[int] | None) -> dict[str, Any]:
        """Apply the chosen fixes (all when ``ids`` is None), then re-audit."""
        from ..fixer import apply_fixes  # noqa: PLC0415

        if self.running:
            raise RuntimeError("an audit is running; wait for it to finish")
        report = self._report
        if report is None:
            raise RuntimeError("no audit has finished yet")
        fixable = report.fixable
        chosen = fixable if ids is None else [fixable[i] for i in ids if 0 <= i < len(fixable)]
        outcomes = apply_fixes(report.target, chosen, dry_run=False)
        result = {
            "applied": [o.as_dict() for o in outcomes if o.applied],
            "refused": [o.as_dict() for o in outcomes if not o.applied],
            "at": time.time(),
        }
        self._update(last_fix=result)
        for outcome in outcomes:
            if outcome.applied:
                self._log(f"fixed: {outcome.finding.title} in {outcome.as_dict()['file']}", "fix")
            else:
                self._log(f"not applied: {outcome.finding.title} - {outcome.reason}", "error")
        # Verify straight away: the history graph shows the before and after.
        self.start_run()
        return result

    # -- serving -----------------------------------------------------------

    def start(self) -> None:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass  # no access log in the user's terminal

            def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionError):
                    self.wfile.write(body)

            def _json(self, payload: Any, status: int = 200) -> None:
                self._send(json.dumps(payload, default=str).encode(), "application/json", status)

            def do_GET(self) -> None:
                route = self.path.split("?", 1)[0]
                if route == "/api/state":
                    self._json(dashboard.snapshot())
                    return
                if route == "/api/report":
                    report = dashboard.snapshot()["report"]
                    if report is None:
                        self._json({"error": "no audit has finished yet"}, 404)
                    else:
                        self._json(report)
                    return
                if route == "/events":
                    self._events()
                    return
                name = "index.html" if route == "/" else Path(route).name
                path = _WEB_DIR / name
                # Basename-only resolution: a traversal attempt cannot escape.
                if not path.is_file() or path.suffix not in _CONTENT_TYPES:
                    self.send_error(404)
                    return
                self._send(path.read_bytes(), _CONTENT_TYPES[path.suffix])

            def _events(self) -> None:
                """Server-sent events: the full state whenever it changes."""
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                version = -1
                try:
                    while not dashboard._stopping.is_set():
                        state = dashboard.snapshot()
                        if state["version"] != version:
                            version = state["version"]
                            data = json.dumps(state, default=str)
                            self.wfile.write(f"data: {data}\n\n".encode())
                            self.wfile.flush()
                        # Wake on change, or every 15s to send a keep-alive.
                        if dashboard.wait_for_change(version, timeout=15) == version:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionError, OSError):
                    return  # the tab closed

            def do_POST(self) -> None:
                # Consume the body before any reply, including refusals: answering
                # while the client is still sending makes Windows reset the
                # connection, so the client sees a dropped socket, not our status.
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                raw = self.rfile.read(min(length, _MAX_BODY)) if length > 0 else b"{}"
                if length > _MAX_BODY:
                    remaining = length - _MAX_BODY
                    while remaining > 0:
                        chunk = self.rfile.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                    self._json({"error": "request too large"}, 413)
                    return

                origin = self.headers.get("Origin")
                host = self.headers.get("Host", "")
                if origin and not origin.rstrip("/").endswith(host):
                    self._json({"error": "cross-origin requests are refused"}, 403)
                    return
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._json({"error": "body is not valid JSON"}, 400)
                    return

                route = self.path.split("?", 1)[0]
                if route == "/api/run":
                    if not dashboard.start_run():
                        self._json({"error": "an audit is already running"}, 409)
                        return
                    self._json({"ok": True})
                    return
                if route == "/api/fix":
                    ids = body.get("ids")
                    if ids is not None and not (
                        isinstance(ids, list) and all(isinstance(i, int) for i in ids)
                    ):
                        self._json({"error": "ids must be a list of integers"}, 400)
                        return
                    try:
                        self._json(dashboard.apply_fixes(ids))
                    except RuntimeError as exc:
                        self._json({"error": str(exc)}, 409)
                    return
                self._json({"error": f"no such endpoint: {route}"}, 404)

        class Server(ThreadingHTTPServer):
            # HTTPServer sets allow_reuse_address = 1, which on Windows lets a
            # second process bind a port that is already serving, so two servers
            # answer the same URL at random instead of the second failing loudly.
            allow_reuse_address = False
            daemon_threads = True

        try:
            self._httpd = Server((self.host, self.port), Handler)
        except OSError as exc:
            raise OSError(
                f"could not start the dashboard on {self.host}:{self.port}: {exc}. "
                "Pass --port to use a different one."
            ) from exc
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            self._changed.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


class ReportServer(LiveDashboard):
    """A dashboard opened on a report that has already finished."""

    def __init__(self, report: AuditReport, host: str = "127.0.0.1", port: int = 8899):
        super().__init__(report.target, host=host, port=port, initial_report=report)
        self._httpd = None


def _block(dashboard: LiveDashboard, open_browser: bool) -> None:
    if open_browser:
        # A browser that will not open never fails a server whose URL is printed.
        with contextlib.suppress(Exception):
            webbrowser.open(dashboard.url)
    try:
        while True:
            time.sleep(0.4)
    except KeyboardInterrupt:
        dashboard.stop()
        print("  dashboard stopped.\n", flush=True)


def _announce(dashboard: LiveDashboard, live: bool) -> None:
    from ..theme import CYAN, detect  # noqa: PLC0415

    theme = detect()
    print()
    print("  " + theme.muted("Dashboard  ") + theme.bold(theme.fg(CYAN, dashboard.url)))
    if live:
        print("  " + theme.muted("The audit is running - watch it live in the browser."))
    print("  " + theme.faint("Ctrl+C to stop the server."))
    print(flush=True)


def serve_live(
    target: Path | str,
    *,
    port: int = 8899,
    host: str = "127.0.0.1",
    offline: bool = False,
    claude_bin: str | None = None,
    open_browser: bool = True,
    block: bool = True,
) -> LiveDashboard:
    """Start the server, print the link, then run the audit in the background."""
    dashboard = LiveDashboard(target, host=host, port=port, offline=offline,
                              claude_bin=claude_bin)
    dashboard.start()
    _announce(dashboard, live=True)
    dashboard.start_run()
    if block:
        _block(dashboard, open_browser)
    return dashboard


def serve_report(
    report: AuditReport,
    *,
    port: int = 8899,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    block: bool = True,
) -> LiveDashboard:
    """Open the dashboard on a finished report (re-runnable from the page)."""
    dashboard = ReportServer(report, host=host, port=port)
    dashboard.start()
    _announce(dashboard, live=False)
    if block:
        _block(dashboard, open_browser)
    return dashboard
