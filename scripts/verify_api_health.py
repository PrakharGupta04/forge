"""End-to-end verification of the Forge API root + health endpoints.

Launches ``uvicorn forge.server.main:app`` on port 8001 as a child
process, waits a few seconds for startup (Postgres pool + Redis client
+ middleware wiring), then probes ``GET /`` and ``GET /health`` with
``httpx``. Both 200 (all components healthy) and 503 (Postgres or
Redis down) are accepted outcomes for ``/health``; the script's job
is to verify the *contract*, not the operational state of the
backing services.

Exit code is 0 on success and non-zero on any assertion failure or
subprocess startup failure. The uvicorn subprocess is always killed in
the ``finally`` block, including when an assertion fires.

Run from the project root::

    python scripts/verify_api_health.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import httpx


_PORT = 8001
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_STARTUP_WAIT_SECONDS = 4.0
_REQUEST_TIMEOUT_SECONDS = 10.0


def main() -> int:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "forge.server.main:app",
            "--port",
            str(_PORT),
            "--host",
            "127.0.0.1",
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        time.sleep(_STARTUP_WAIT_SECONDS)

        # Verify the process is still alive — if uvicorn crashed during
        # startup, surface its captured output rather than failing on a
        # confusing httpx ConnectError below.
        if proc.poll() is not None:
            captured = (
                proc.stdout.read().decode("utf-8", errors="replace")
                if proc.stdout
                else ""
            )
            print("uvicorn subprocess exited during startup:")
            print(captured)
            return 1

        with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            root_resp = client.get(f"{_BASE_URL}/")
            health_resp = client.get(f"{_BASE_URL}/health")

        # ---- GET / ----
        assert root_resp.status_code == 200, (
            f"GET / expected 200, got {root_resp.status_code}: {root_resp.text}"
        )
        root_body = root_resp.json()
        assert root_body.get("name") == "Forge", (
            f"GET / missing or wrong 'name': {root_body!r}"
        )
        assert root_body.get("version") == "0.1.0", (
            f"GET / missing or wrong 'version': {root_body!r}"
        )

        # ---- GET /health ----
        # Both 200 (all components healthy) and 503 (Postgres or Redis
        # down) are valid: the verify script asserts the contract, not
        # the operational state of the backing services.
        assert health_resp.status_code in (200, 503), (
            f"GET /health expected 200 or 503, got "
            f"{health_resp.status_code}: {health_resp.text}"
        )
        health_body = health_resp.json()
        assert "components" in health_body, (
            f"GET /health missing 'components' key: {health_body!r}"
        )
        components = health_body["components"]
        for required_key in ("postgresql", "redis", "embedding_model"):
            assert required_key in components, (
                f"GET /health components missing {required_key!r}: "
                f"{components!r}"
            )
            assert isinstance(components[required_key], bool), (
                f"GET /health components[{required_key!r}] must be bool, "
                f"got {type(components[required_key]).__name__}: "
                f"{components[required_key]!r}"
            )
        assert health_body.get("status") in ("healthy", "degraded"), (
            f"GET /health 'status' must be 'healthy' or 'degraded': "
            f"{health_body!r}"
        )
        assert health_body.get("version") == "0.1.0", (
            f"GET /health 'version' must be '0.1.0': {health_body!r}"
        )
        assert "timestamp" in health_body, (
            f"GET /health missing 'timestamp': {health_body!r}"
        )

        print(f"GET /         -> {root_resp.status_code}")
        print(json.dumps(root_body, indent=2))
        print()
        print(f"GET /health   -> {health_resp.status_code}")
        print(json.dumps(health_body, indent=2))
        print()
        print("API health endpoint verified")
        return 0

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    sys.exit(main())
