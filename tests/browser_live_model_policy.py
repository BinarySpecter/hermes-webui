#!/usr/bin/env python3
"""
Headless browser regression for the live-model discovered-vs-pinned policy.

WHY THIS EXISTS
  `static/ui.js` used to keep a browser response cache for `/api/models/live`,
  keyed by profile + provider. That cache could replay a broad *discovered*
  catalog after the same profile switched to a strict model pin, because the
  client copy never consulted the policy-keyed server cache. The server fix
  (60s TTL keyed by profile, provider, and a discovered-vs-pinned policy
  fingerprint) cannot help if the browser short-circuits on a local hit.
  #7404 review removed the browser cache; this gate locks that in behaviourally.

  It is a regression guard, not a unit test: it boots the real `server.py`
  agent-free and drives the real `populateModelDropdown()` entry point, stubbing
  `/api/models/live` with Playwright so no provider, credential, or agent is
  needed. Two assertions are the required regressions:
    1. Same-profile discovered -> strict pin: the stale unpinned model must not
       be re-applied, AND the endpoint must actually be re-requested (proving
       the browser did not short-circuit).
    2. Profile switch: a catalog served for profile A must not be applied when
       the active profile is B.
  A third case exercises the in-flight profile re-verification: a response for
  the profile captured at fetch start must be dropped if the profile changed
  before it was applied.
  A fourth case exercises the reference-counted pending entry (#7404 review):
  two overlapping fetches for the same profile+provider share one key, and the
  key must stay pending until the older one has resolved and only clear when the
  last one resolves. The route handler is put in "hold" mode so response timing
  is released deterministically from the test (never with a blocking sleep).

USAGE
  python tests/browser_live_model_policy.py
  (Requires: playwright + chromium. Boots server.py on an ephemeral port with an
  isolated temp state dir and no agent.)

EXIT CODES
  0 — all policy regressions held
  1 — a regression was observed (stale model re-applied, no re-fetch, or
      cross-profile leak)
  2 — environment/setup failure (server didn't boot, playwright missing, etc.)
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# 'openrouter' avoids the `@provider:` ID prefixing `_addLiveModelsToSelect`
# applies to portal-style providers, keeping option values easy to assert.
PROVIDER = "openrouter"

BROAD_DISCOVERED = [
    {"id": "broad-live-1", "label": "Broad Live 1"},
    {"id": "broad-live-2", "label": "Broad Live 2"},
    {"id": "broad-live-3", "label": "Broad Live 3"},
]
STRICT_PIN = [{"id": "strict-pin-1", "label": "Strict Pin 1"}]
PROFILE_A_CATALOG = [{"id": "a-live-1", "label": "A Live 1"}]
PROFILE_B_CATALOG = [{"id": "b-live-1", "label": "B Live 1"}]
INFLIGHT_CATALOG = [{"id": "inflight-live-1", "label": "Inflight Live 1"}]

PROFILE_A = "policy-profile-a"
PROFILE_B = "policy-profile-b"
PROFILE_INFLIGHT = "policy-profile-inflight"
PROFILE_INFLIGHT_RACED = "policy-profile-inflight-raced"
PROFILE_CONCURRENT = "policy-profile-concurrent"
CONCURRENT_CATALOG = [{"id": "concurrent-live-1", "label": "Concurrent Live 1"}]

BENIGN = [
    "favicon",
    "manifest.json",
    "serviceworker",
    "sw.js",
    "the server responded with a status of 404",
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout: float = 30.0, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    return False


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _option_values(page) -> list[str]:
    return page.evaluate(
        "Array.from(document.querySelectorAll('#modelSelect option')).map(o => o.value)"
    )


def _wait_for_option(page, model_id: str, timeout: float = 8000.0) -> bool:
    try:
        page.wait_for_function(
            "id => Array.from(document.querySelectorAll('#modelSelect option'))"
            ".some(o => o.value === id)",
            arg=model_id,
            timeout=timeout,
        )
        return True
    except Exception:
        return False


def _wait_for_live_requests(page, count: int, timeout: float = 8000.0) -> bool:
    """Wait for the browser to issue more than *count* live-model fetches.

    MUST go through a Playwright call (``wait_for_function``) rather than a
    Python ``time.sleep`` busy-wait: route handlers are dispatched on
    Playwright's event loop, which only advances while a Playwright API call is
    pumping. A pure-Python poll never lets the intercepted request be fulfilled,
    so the counter would never move and the wait would hang until the CI job
    timeout. ``wait_for_function`` also measures the app's own fetch call, which
    is exactly the "did the browser really re-request?" property under test.
    """
    try:
        page.wait_for_function(
            "n => (window.__liveFetchCount || 0) > n",
            arg=count,
            timeout=timeout,
        )
        return True
    except Exception:
        return False


def _wait_for_js(page, expression: str, arg=None, timeout: float = 8000.0) -> bool:
    """Return True once *expression* is truthy in the page, else False.

    ``wait_for_function`` raises on timeout; this normalises that to a bool and
    always pumps Playwright's event loop until the condition holds.
    """
    try:
        page.wait_for_function(expression, arg=arg, timeout=timeout)
        return True
    except Exception:
        return False


def _wait_for_held_routes(stub: "LiveModelStub", count: int, page, timeout: float = 8000.0) -> bool:
    """Pump Playwright until *count* live routes are held by the stub.

    Uses ``page.wait_for_timeout`` so each iteration advances Playwright's event
    loop (which is what actually dispatches route handlers); a Python
    ``time.sleep`` would deadlock the interception.
    """
    deadline = time.time() + timeout
    while len(stub.held_routes) < count:
        if time.time() > deadline:
            return False
        page.wait_for_timeout(25)
    return True


def _capture_page_errors(page) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []

    def on_console(message):
        if message.type != "error":
            return
        text = message.text
        if not any(needle in text.lower() for needle in BENIGN):
            errors.append(("console", text))

    page.on("console", on_console)
    page.on("pageerror", lambda error: errors.append(("pageerror", str(error))))
    return errors


class LiveModelStub:
    """Serves deterministic `/api/models` and `/api/models/live` payloads."""

    def __init__(self) -> None:
        self.active_provider = PROVIDER
        self.live_models: list[dict] = []
        self.live_request_count = 0
        self.live_requests: list[str] = []
        # When set, live responses are held instead of fulfilled so the test can
        # release overlapping requests one at a time, deterministically.
        self.hold_live = False
        self.held_routes: list = []
        self.fulfilled_count = 0

    @staticmethod
    def _is_live_url(url: str) -> bool:
        return url.split("?", 1)[0].endswith("/api/models/live")

    def _models_payload(self) -> dict:
        return {
            "active_provider": self.active_provider,
            "default_model": "static-base",
            "configured_model_badges": {},
            "groups": [
                {
                    "provider": "OpenRouter",
                    "provider_id": self.active_provider,
                    "models": [{"id": "static-base", "label": "Static Base"}],
                }
            ],
        }

    def _live_payload(self) -> dict:
        return {"provider": self.active_provider, "models": self.live_models}

    def handle_live(self, route) -> None:
        self.live_request_count += 1
        self.live_requests.append(route.request.url)
        if self.hold_live:
            self.held_routes.append(route)
            return
        self.release_live(route)

    def release_live(self, route) -> None:
        self.fulfilled_count += 1
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(self._live_payload()),
        )

    def handle_models(self, route) -> None:
        if self._is_live_url(route.request.url):
            self.handle_live(route)
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(self._models_payload()),
        )


def _install_flip_on_live_fetch(page) -> None:
    """Test harness hook: flip the active profile when the next live fetch runs.

    Runs in the page before app scripts. `_fetchLiveModels()` captures the
    profile before awaiting `fetch`; this hook changes it synchronously inside
    that call so the captured value and the post-await value differ.
    """
    page.add_init_script(
        """
        (() => {
          const realFetch = window.fetch.bind(window);
          window.__flipProfileOnNextLiveFetch = null;
          window.__liveFetchCount = 0;
          window.fetch = function(input, init) {
            const url = (typeof input === 'string') ? input : (input && input.url) || '';
            if (url.includes('/api/models/live')) {
              window.__liveFetchCount += 1;
              if (window.__flipProfileOnNextLiveFetch) {
                const target = window.__flipProfileOnNextLiveFetch;
                window.__flipProfileOnNextLiveFetch = null;
                S.activeProfile = target;
              }
            }
            return realFetch(input, init);
          };
        })();
        """
    )


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SETUP FAIL: playwright is not installed", file=sys.stderr)
        return 2

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_py = os.path.join(repo_root, "server.py")
    if not os.path.exists(server_py):
        print(f"SETUP FAIL: server.py not found at {server_py}", file=sys.stderr)
        return 2

    port = int(os.getenv("HERMES_LIVE_MODEL_POLICY_PORT", "") or _free_port())
    base_url = f"http://127.0.0.1:{port}"
    state_dir = tempfile.mkdtemp(prefix="hermes-live-model-policy-")
    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY"):
            env.pop(key, None)
    env.update({
        "HERMES_WEBUI_PORT": str(port),
        "HERMES_WEBUI_HOST": "127.0.0.1",
        "HERMES_WEBUI_STATE_DIR": state_dir,
        "HERMES_HOME": state_dir,
        "HERMES_BASE_HOME": state_dir,
        "HERMES_WEBUI_SKIP_ONBOARDING": "1",
        "HERMES_WEBUI_AGENT_DIR": os.path.join(state_dir, "no-agent"),
    })

    log_path = os.path.join(state_dir, "server.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, server_py],
        cwd=repo_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}),
    )
    browser = None
    playwright = None
    try:
        if not _wait_for_health(base_url, timeout=30, proc=proc):
            print("SETUP FAIL: server did not become healthy in 30s", file=sys.stderr)
            log.flush()
            with open(log_path) as handle:
                print(handle.read()[-2000:], file=sys.stderr)
            return 2

        stub = LiveModelStub()
        stub.live_models = list(BROAD_DISCOVERED)
        errors: list[tuple[str, str]] = []
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(base_url=base_url)
        page = context.new_page()
        _install_flip_on_live_fetch(page)
        page.route("**/api/models**", stub.handle_models)
        page.route("**/api/models/live**", stub.handle_live)
        errors = _capture_page_errors(page)
        page.goto("/", wait_until="domcontentloaded")
        page.wait_for_selector("#modelSelect", state="attached", timeout=15000)

        # Seed the dropdown with the broad discovered catalog (the app's own
        # boot hydration does this; wait for it to land).
        if not _wait_for_option(page, "broad-live-1"):
            raise AssertionError(
                "setup: discovered catalog was never applied on boot; "
                f"observed options={_option_values(page)!r}; "
                f"live requests={stub.live_request_count}; errors={errors!r}"
            )
        print("OK  discovered catalog applied:", _option_values(page))
        baseline_requests = stub.live_request_count

        # --- Regression 1: same-profile discovered -> strict pin --------------
        stub.live_models = list(STRICT_PIN)
        page.evaluate("void populateModelDropdown()")
        applied_pin = _wait_for_option(page, "strict-pin-1")
        ids_after_pin = _option_values(page)
        if not applied_pin:
            raise AssertionError(
                "same-profile discovered->pin: strict pin 'strict-pin-1' was never "
                f"applied; observed options={ids_after_pin!r}; "
                f"live requests before={baseline_requests} after={stub.live_request_count}"
            )
        if stub.live_request_count <= baseline_requests:
            raise AssertionError(
                "same-profile discovered->pin: browser did not re-request "
                f"/api/models/live after the policy change; observed requests="
                f"{stub.live_request_count} (baseline={baseline_requests}); "
                f"options={ids_after_pin!r}"
            )
        if "broad-live-1" in ids_after_pin:
            raise AssertionError(
                "same-profile discovered->pin: stale unpinned model 'broad-live-1' "
                f"was re-applied; observed options={ids_after_pin!r}"
            )
        print(
            "OK  same-profile pin replaced discovered catalog:",
            ids_after_pin,
            f"(live requests {baseline_requests}->{stub.live_request_count})",
        )

        # --- Regression 2: profile switch ------------------------------------
        page.evaluate(f"S.activeProfile = {PROFILE_A!r}")
        stub.live_models = list(PROFILE_A_CATALOG)
        page.evaluate("void populateModelDropdown()")
        if not _wait_for_option(page, "a-live-1"):
            raise AssertionError(
                "profile switch: catalog for profile A was never applied; "
                f"observed options={_option_values(page)!r}"
            )
        page.evaluate(f"S.activeProfile = {PROFILE_B!r}")
        stub.live_models = list(PROFILE_B_CATALOG)
        page.evaluate("void populateModelDropdown()")
        applied_b = _wait_for_option(page, "b-live-1")
        ids_after_switch = _option_values(page)
        if not applied_b:
            raise AssertionError(
                "profile switch: catalog for profile B was never applied; "
                f"observed options={ids_after_switch!r}"
            )
        if "a-live-1" in ids_after_switch:
            raise AssertionError(
                "profile switch: profile A catalog 'a-live-1' leaked into profile "
                f"B; observed options={ids_after_switch!r}"
            )
        print("OK  profile switch dropped the previous profile catalog:", ids_after_switch)

        # --- Case 3: in-flight profile re-verification -----------------------
        page.evaluate(f"S.activeProfile = {PROFILE_INFLIGHT!r}")
        stub.live_models = list(INFLIGHT_CATALOG)
        before_inflight = page.evaluate("window.__liveFetchCount || 0")
        page.evaluate(f"window.__flipProfileOnNextLiveFetch = {PROFILE_INFLIGHT_RACED!r}")
        page.evaluate("void populateModelDropdown()")
        if not _wait_for_live_requests(page, before_inflight):
            raise AssertionError(
                "in-flight profile check: /api/models/live was never requested; "
                f"observed requests={stub.live_request_count}"
            )
        page.wait_for_function(
            "profile => typeof S !== 'undefined' && S.activeProfile === profile",
            arg=PROFILE_INFLIGHT_RACED,
            timeout=5000,
        )
        page.wait_for_timeout(500)
        ids_after_inflight = _option_values(page)
        if "inflight-live-1" in ids_after_inflight:
            raise AssertionError(
                "in-flight profile check: catalog captured for profile "
                f"{PROFILE_INFLIGHT!r} was applied after the active profile changed "
                f"to {PROFILE_INFLIGHT_RACED!r}; observed options={ids_after_inflight!r}"
            )
        print("OK  in-flight response dropped after profile changed:", ids_after_inflight)

        # --- Case 4: overlapping requests must keep the pending key alive -----
        #
        # syncTopbar() defers a model correction while a fetch is pending, so the
        # observable requirement is: while ANY request for a key is in flight,
        # _liveModelFetchPending.has(key) must be true. This case asserts on
        # .has() only — .has() exists on both a Set (the pre-fix shape, which
        # cleared the key as soon as ANY request finished) and the current
        # reference-counted Map. That way it fails pre-fix for the RACE rather
        # than for an API mismatch (e.g. .get is not a function on a Set).
        page.evaluate(f"S.activeProfile = {PROFILE_CONCURRENT!r}")
        stub.live_models = list(CONCURRENT_CATALOG)
        pending_key = page.evaluate(f"() => _liveModelFetchKey({PROVIDER!r})")
        # Hold live responses so both fetches are genuinely in flight at once.
        # They are released explicitly below; no blocking sleep is used.
        stub.hold_live = True
        stub.held_routes = []
        page.evaluate(
            "() => {"
            "  const sel = document.getElementById('modelSelect');"
            f"  void _fetchLiveModels({PROVIDER!r}, sel);"
            f"  void _fetchLiveModels({PROVIDER!r}, sel);"
            "}"
        )
        if not _wait_for_held_routes(stub, 2, page):
            raise AssertionError(
                "concurrency: expected two overlapping /api/models/live "
                f"requests to be held; held={len(stub.held_routes)} "
                f"total={stub.live_request_count}"
            )
        if not page.evaluate("key => _liveModelFetchPending.has(key)", pending_key):
            raise AssertionError(
                "concurrency: key must be pending while both requests are in flight"
            )

        # Release ONLY the older response. Its completion is observable: the
        # catalog it carries reaches the dropdown. The newer request is still
        # held, so the key MUST still be pending afterwards.
        stub.release_live(stub.held_routes[0])
        if not _wait_for_option(page, "concurrent-live-1"):
            raise AssertionError(
                "concurrency: the first (older) request never completed after "
                f"being released; options={_option_values(page)!r}"
            )
        if not page.evaluate("key => _liveModelFetchPending.has(key)", pending_key):
            raise AssertionError(
                "concurrency: pending entry cleared after the FIRST of two "
                "overlapping requests completed while a newer request for the same "
                "profile+provider is still in flight — syncTopbar() would now "
                "persist a static fallback over the session's intended model"
            )
        print("OK  pending key survived the first of two overlapping requests")

        # Release the last response; only now may the key clear.
        stub.release_live(stub.held_routes[1])
        if not _wait_for_js(
            page, "key => !_liveModelFetchPending.has(key)", pending_key
        ):
            raise AssertionError(
                "concurrency: pending entry did not clear after the LAST "
                "overlapping request completed; still pending"
            )
        print("OK  pending key cleared after the last overlapping request")

        if errors:
            raise AssertionError(f"unexpected browser errors: {errors!r}")
        print("\nLIVE MODEL POLICY GATE PASSED")
        return 0
    except Exception as error:
        print(f"\nLIVE MODEL POLICY GATE FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()
        _terminate_process(proc)
        log.close()


if __name__ == "__main__":
    sys.exit(main())
