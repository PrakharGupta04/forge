"""Smoke-verify the Forge Docker stack: postgres + redis + api.

This script is the post-``docker compose up`` health probe. It does not
manage the stack lifecycle (no ``up``/``down``); it inspects whatever
compose project is currently running and asserts the contracts the API
exposes against the live containers.

Checks performed
----------------
1. ``docker compose ps`` lists ``postgres`` and ``redis`` services in a
   running state. The check accepts both the modern ``docker compose``
   (Compose v2 plugin) and the legacy ``docker-compose`` binary, falling
   back to plain ``docker ps`` if neither compose CLI is installed.
2. ``GET http://localhost:8000/health`` returns either ``200`` (all
   components healthy) or ``503`` (one or more degraded). Both are valid
   contractually: the API is designed to start and respond even when
   PostgreSQL or Redis is unavailable, so the lifespan can never get
   wedged at boot. The probe asserts only that the response is a
   well-formed ``HealthResponse`` with the documented component map.

Exit codes
----------
* ``0`` — both checks pass
* ``1`` — any check fails (mis-printed services, network error,
  malformed response, etc.); the failing reason is printed
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Optional

import httpx


_API_HEALTH_URL = "http://localhost:8000/health"
_REQUIRED_SERVICES = ("postgres", "redis")
_HEALTH_TIMEOUT_SECONDS = 10.0


def _run(cmd: list[str]) -> Optional[subprocess.CompletedProcess]:
    """Run ``cmd`` and return the result, or ``None`` if the binary is missing.

    Surfaces transport-level CLI failures (binary not on PATH) as
    ``None`` so the caller can try a fallback, while letting non-zero
    exit codes through as a populated ``CompletedProcess`` so the caller
    can inspect ``stderr`` / ``returncode``.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None


def _compose_ps_json() -> Optional[list[dict]]:
    """Return ``docker compose ps`` rows as a list of dicts.

    Tries ``docker compose ps`` first (Compose v2 plugin, the default on
    modern Docker), then ``docker-compose ps`` for older installs. On
    success the JSON output is parsed; on failure ``None`` is returned
    and the caller falls back to a plain ``docker ps`` inspection.
    """
    for cmd in (
        ["docker", "compose", "ps", "--format", "json"],
        ["docker-compose", "ps", "--format", "json"],
    ):
        result = _run(cmd)
        if result is None or result.returncode != 0:
            continue
        stdout = result.stdout.strip()
        if not stdout:
            return []
        # Newer compose prints one JSON object per line; older prints a
        # JSON array. Handle both without guessing the format.
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            rows: list[dict] = []
            for line in stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    return None
            return rows
    return None


def _services_running_via_compose() -> Optional[dict[str, str]]:
    """Map service name -> state string from compose, or ``None`` if unavailable."""
    rows = _compose_ps_json()
    if rows is None:
        return None
    states: dict[str, str] = {}
    for row in rows:
        service = row.get("Service") or row.get("Name") or ""
        state = (
            row.get("State")
            or row.get("Status")
            or row.get("Health")
            or "unknown"
        )
        if service:
            states[service] = str(state)
    return states


def _services_running_via_docker_ps() -> dict[str, str]:
    """Fallback: parse ``docker ps`` for containers whose names match services.

    Used when no compose CLI is installed (rare on CI but possible on
    stripped-down hosts). Matches by substring on the container name so
    ``forge-postgres-1``, ``forge_postgres_1``, etc. all resolve.
    """
    result = _run(
        ["docker", "ps", "--format", "{{.Names}}|{{.Status}}"]
    )
    states: dict[str, str] = {}
    if result is None or result.returncode != 0:
        return states
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, status = line.split("|", 1)
        for svc in _REQUIRED_SERVICES:
            if svc in name and svc not in states:
                states[svc] = status
    return states


def _check_services() -> bool:
    states = _services_running_via_compose()
    if states is None or not any(
        svc in states for svc in _REQUIRED_SERVICES
    ):
        # No compose info, or compose returned an empty/unrelated set —
        # fall through to plain ``docker ps`` so this script is still
        # useful on hosts where compose is unavailable.
        states = _services_running_via_docker_ps()

    print("Compose services:")
    if not states:
        print("  (no containers reported — is the stack started?)")
        return False
    for svc, state in sorted(states.items()):
        print(f"  {svc:>10}: {state}")

    missing = [svc for svc in _REQUIRED_SERVICES if svc not in states]
    if missing:
        print(f"FAIL: required services not running: {missing}")
        return False

    not_running = [
        svc
        for svc in _REQUIRED_SERVICES
        # Accept any state string that indicates "up"; compose v2 reports
        # ``"running"``, docker ps reports ``"Up X seconds"``.
        if "running" not in states[svc].lower()
        and not states[svc].lower().startswith("up")
    ]
    if not_running:
        print(
            f"FAIL: required services not in a running state: {not_running}"
        )
        return False

    return True


def _check_health() -> bool:
    """Probe ``GET /health`` and print the component map.

    Both 200 (healthy) and 503 (degraded) are accepted: the API contract
    explicitly allows the process to start with one or more components
    unreachable so the health endpoint can report the degraded state
    rather than the whole container crashing. The probe only fails if
    the response status is something else, the response body cannot be
    parsed, or the request itself errors (connection refused, timeout).
    """
    try:
        with httpx.Client(timeout=_HEALTH_TIMEOUT_SECONDS) as client:
            resp = client.get(_API_HEALTH_URL)
    except httpx.HTTPError as exc:
        print(f"FAIL: GET {_API_HEALTH_URL} raised {type(exc).__name__}: {exc}")
        return False

    print(f"GET /health -> {resp.status_code}")
    if resp.status_code not in (200, 503):
        print(f"FAIL: unexpected status code; body was: {resp.text}")
        return False

    try:
        body = resp.json()
    except ValueError:
        print(f"FAIL: /health response was not JSON: {resp.text!r}")
        return False

    components = body.get("components")
    if not isinstance(components, dict):
        print(f"FAIL: /health response missing components map: {body!r}")
        return False

    print("Component health:")
    for key in ("postgresql", "redis", "embedding_model"):
        if key not in components:
            print(f"FAIL: components map missing {key!r}: {components!r}")
            return False
        print(f"  {key:>16}: {components[key]}")

    notes = body.get("notes")
    if isinstance(notes, dict):
        for k, v in notes.items():
            print(f"  note ({k}): {v}")

    return True


def main() -> int:
    services_ok = _check_services()
    print()
    health_ok = _check_health()
    print()
    if services_ok and health_ok:
        print("Docker stack verified")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
