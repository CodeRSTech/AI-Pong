import functools
import json
import os
import shutil
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

from src.components.geometry import Vec2
from src.components.playzone import PlayZone
from src.ga.player import IndividualPlayer


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def browser():
    candidates = [
        os.environ.get("PONG_TEST_BROWSER"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("Browser integration requires an installed Chromium-family browser.")


@pytest.fixture
def web_site(tmp_path):
    shutil.copytree(ROOT / "assets", tmp_path / "assets")
    shutil.copyfile(ROOT / "index.html", tmp_path / "index.html")

    result_ready = threading.Event()
    results = []

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.path != "/__test_result":
                self.send_error(404)
                return
            results.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()
            result_ready.set()

    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f"http://127.0.0.1:{server.server_port}", result_ready, results
    server.shutdown()
    server.server_close()
    thread.join()


def run_browser(browser, web_site, script, *, homepage=False, flags=()):
    directory, url, result_ready, results = web_site
    prefix = (directory / "index.html").read_text(encoding="utf-8") if homepage else (
        '<html><body><script src="./assets/pong-core.js"></script></body></html>'
    )
    harness = """
    <script>
    window.addEventListener("load", async () => {
      let result;
      try {
        const answer = await (async () => { SCRIPT })();
        result = {ok: true, answer};
      } catch (error) {
        result = {ok: false, error: error.stack};
      }
      await fetch("/__test_result", {method: "POST", body: JSON.stringify(result)});
    });
    </script>
    """.replace("SCRIPT", script)
    (directory / "test.html").write_text(prefix.replace("</body>", harness + "</body>"), encoding="utf-8")
    # Use real time: Chromium's dump-DOM virtual clock does not advance animation frames.
    process = subprocess.Popen(
        [browser, "--headless=new", "--disable-gpu", "--no-first-run",
         f"--user-data-dir={directory / 'browser-profile'}",
         "--remote-debugging-port=0", *flags, f"{url}/test.html"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert result_ready.wait(30), f"Browser did not report a result (exit={process.poll()})."
        result = results[0]
    finally:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {process.pid}"],
                    check=True, capture_output=True,
                )
            else:
                process.terminate()
            process.wait(timeout=10)
    assert result["ok"], result.get("error")
    return result["answer"]


def snapshot(zone):
    return {
        "ball": {"x": zone.ball.pos_x, "y": zone.ball.pos_y,
                 "vx": zone.ball.speed.x, "vy": zone.ball.speed.y, "width": 12, "height": 12},
        "ai": {"x": zone.ai_paddle.pos_x, "y": zone.ai_paddle.pos_y, "width": 80, "height": 10},
        "cpu": {"x": zone.cpu_paddle.pos_x, "y": zone.cpu_paddle.pos_y, "width": 80, "height": 10},
        "scores": {"ai": zone.ai_player.scores["Player"], "cpu": zone.ai_player.scores["CPU"]},
    }


def test_browser_inference_and_physics_match_python(browser, web_site):
    player = IndividualPlayer()
    player.neural_net.load_state_dict(torch.load(ROOT / "assets/model.pt", map_location="cpu", weights_only=True))
    observations = np.random.default_rng(14).uniform(-1, 1, (64, 7)).tolist()
    observations.extend([[0.0] * 7, [1.0] * 7, [-1.0] * 7])
    with torch.no_grad():
        expected = player.neural_net(torch.tensor(observations, dtype=torch.float32)).numpy()

    scenarios = []
    for x, y, vx, vy in [(200, 250, 1, 4), (3, 250, -1, 4), (230, 487, 1, 4),
                         (245, 10, 1, -4), (200, 499, 1, 4), (200, 1, 1, -4)]:
        player.reset_scores()
        zone = PlayZone(476, 500, 2.5, player)
        zone.ball.pos_x, zone.ball.pos_y = x, y
        zone.ball.speed = Vec2(vx, vy)

        def fixed_serve(ball):
            ball.pos_x, ball.pos_y = 238, 250
            ball.speed = Vec2(1.1, 3.9)

        zone.respawn_ball = fixed_serve
        before = snapshot(zone)
        observed = player.look(zone).tolist()
        actions = player.think(observed)[0]
        player.apply_move(zone, bool(actions[0]), bool(actions[1]))
        zone.update()
        scenarios.append({"before": before, "observed": observed, "after": snapshot(zone)})

    script = """
      const core = PongCore;
      const model = core.validateModel(await (await fetch("./assets/model.json")).json());
      const observations = OBSERVATIONS;
      const scenarios = SCENARIOS;
      const inference = observations.map(values => core.predict(model, values));
      const physics = scenarios.map(scenario => {
        const state = structuredClone(scenario.before);
        const observed = core.observe(state);
        core.step(state, model, () => 0.5);
        return {observed, state};
      });
      function assert(value, message) { if (!value) throw new Error(message); }
      const invalid = structuredClone(model);
      invalid.layers[0].weight[0].pop();
      let rejected = false;
      try { core.validateModel(invalid); } catch (_) { rejected = true; }
      assert(rejected, "Malformed model was accepted");
      const zero = structuredClone(model);
      zero.layers.forEach(layer => {
        layer.weight.forEach(row => row.fill(0)); layer.bias.fill(0);
      });
      assert(core.predict(zero, Array(7).fill(0)).actions.every(v => !v), "Threshold must be strict");
      const state = core.createState(() => 0.5);
      for (const actions of [[false, false], [true, true]]) {
        const before = state.ai.x; core.moveAI(state, actions);
        assert(state.ai.x === before, "Conflicting/absent actions must not move");
      }
      state.ai.x = 40; core.moveAI(state, [true, false]);
      assert(state.ai.x === 40, "AI left bound");
      state.ai.x = 436; core.moveAI(state, [false, true]);
      assert(state.ai.x === 436, "AI right bound");
      const position = core.displayPosition({x: 238, y: 495});
      assert(position.left === 99 && position.top === 50, "Rotated placement");
      const a = core.createState(() => 0.5), b = structuredClone(a);
      let ra = 0, rb = 0;
      for (let i = 0; i < 60; i++) ra = core.advance(a, model, 1/60, ra, () => 0.5);
      for (let i = 0; i < 144; i++) rb = core.advance(b, model, 1/144, rb, () => 0.5);
      assert(JSON.stringify(a) === JSON.stringify(b), "Refresh rate changed simulation");
      const c = core.createState(() => 0.5), d = structuredClone(c);
      core.advance(c, model, 60, 0, () => 0.5);
      core.advance(d, model, 0.1, 0, () => 0.5);
      assert(JSON.stringify(c) === JSON.stringify(d), "Unbounded catch-up");
      return {inference, physics};
    """.replace("OBSERVATIONS", json.dumps(observations)).replace("SCENARIOS", json.dumps(scenarios))
    result = run_browser(browser, web_site, script)
    np.testing.assert_allclose([entry["raw"] for entry in result["inference"]], expected, atol=2e-6, rtol=2e-6)
    assert [entry["actions"] for entry in result["inference"]] == (expected > 0.5).tolist()
    for scenario, actual in zip(scenarios, result["physics"]):
        np.testing.assert_allclose(actual["observed"], scenario["observed"], atol=1e-12)
        for entity in ("ball", "ai", "cpu", "scores"):
            for name, value in scenario["after"][entity].items():
                assert actual["state"][entity][name] == pytest.approx(value, abs=1e-10)


@pytest.mark.parametrize("reduced", [False, True])
def test_homepage_loads_and_pause_resume_work(browser, web_site, reduced):
    script = """
      const demo = document.querySelector("[data-pong-demo]");
      const status = demo.querySelector("[data-pong-status]");
      const button = demo.querySelector("button");
      const wait = () => new Promise(resolve => setTimeout(resolve, 300));
      await wait();
      if (button.disabled) throw new Error("Model did not load: " + status.textContent);
      const initialStatus = status.textContent;
      if (initialStatus !== EXPECTED) throw new Error("Unexpected startup: " + initialStatus);
      if (initialStatus === "Live model") button.click();
      if (status.textContent !== "Paused") throw new Error("Pause failed");
      const before = demo.querySelector(".ball").getAttribute("style");
      await wait();
      if (before !== demo.querySelector(".ball").getAttribute("style")) throw new Error("Paused game moved");
      button.click();
      if (status.textContent !== "Live model") throw new Error("Resume failed");
      const resumed = demo.querySelector(".ball").getAttribute("style");
      await wait();
      if (resumed === demo.querySelector(".ball").getAttribute("style")) throw new Error("Live game did not animate");
      const court = demo.querySelector(".court").getBoundingClientRect();
      const ai = demo.querySelector(".paddle.right").getBoundingClientRect();
      if (ai.left < court.left || ai.right > court.right + 1) throw new Error("Paddle placement");
      const visibility = Object.getOwnPropertyDescriptor(document, "hidden");
      Object.defineProperty(document, "hidden", {configurable: true, value: true});
      document.dispatchEvent(new Event("visibilitychange"));
      if (status.textContent !== "Suspended") throw new Error("Hidden game did not suspend");
      const suspended = demo.querySelector(".ball").getAttribute("style");
      await wait();
      if (suspended !== demo.querySelector(".ball").getAttribute("style")) throw new Error("Suspended game moved");
      button.click();
      if (status.textContent !== "Paused") throw new Error("Manual pause while hidden failed");
      if (visibility) Object.defineProperty(document, "hidden", visibility);
      else delete document.hidden;
      document.dispatchEvent(new Event("visibilitychange"));
      if (status.textContent !== "Paused") throw new Error("Visibility discarded manual pause");
      button.click();
      window.scrollTo({top: document.body.scrollHeight, behavior: "instant"});
      await wait();
      if (status.textContent !== "Suspended") throw new Error("Offscreen game did not suspend");
      demo.scrollIntoView({behavior: "instant"});
      await wait();
      if (status.textContent !== "Live model") throw new Error("Visible game did not resume");
      return {initialStatus, button: button.textContent};
    """.replace("EXPECTED", json.dumps("Paused" if reduced else "Live model"))
    flags = ["--force-prefers-reduced-motion", "--window-size=390,900"] if reduced else []
    result = run_browser(browser, web_site, script, homepage=True, flags=flags)
    assert result["button"] == "Pause demo"


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_homepage_surfaces_model_failure(browser, web_site, failure):
    directory = web_site[0]
    model = directory / "assets/model.json"
    if failure == "missing":
        model.unlink()
    else:
        model.write_text('{"version": 99}')
    result = run_browser(browser, web_site, """
      await new Promise(resolve => setTimeout(resolve, 300));
      const demo = document.querySelector("[data-pong-demo]");
      return {status: demo.querySelector("[data-pong-status]").textContent,
        disabled: demo.querySelector("button").disabled,
        errorVisible: !demo.querySelector("[data-pong-error]").hidden};
    """, homepage=True)
    assert result == {"status": "Demo unavailable", "disabled": True, "errorVisible": True}
