(function () {
  "use strict";

  const core = window.PongCore;
  const demo = document.querySelector("[data-pong-demo]");
  const status = demo.querySelector("[data-pong-status]");
  const button = demo.querySelector("[data-pong-toggle]");
  const aiScore = demo.querySelector("[data-pong-ai-score]");
  const cpuScore = demo.querySelector("[data-pong-cpu-score]");
  const ballElement = demo.querySelector(".ball");
  const aiElement = demo.querySelector(".paddle.right");
  const cpuElement = demo.querySelector(".paddle.left");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  status.textContent = "Loading model";
  let paused = motion.matches;
  let visible = false;
  let model;
  let state;
  let frame = null;
  let previous = null;
  let remainder = 0;
  let failed = false;

  function render() {
    demo.querySelector(".court").classList.add("live");
    for (const [element, entity] of [[ballElement, state.ball], [aiElement, state.ai], [cpuElement, state.cpu]]) {
      const position = core.displayPosition(entity);
      element.style.left = `${position.left}%`;
      element.style.top = `${position.top}%`;
      element.style.width = `${entity.height / core.HEIGHT * 100}%`;
      element.style.height = `${entity.width / core.WIDTH * 100}%`;
    }
    aiScore.textContent = state.scores.ai;
    cpuScore.textContent = state.scores.cpu;
  }

  function fail(error) {
    failed = true;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    button.disabled = true;
    status.textContent = "Demo unavailable";
    demo.querySelector("[data-pong-error]").hidden = false;
    console.error("AI-Pong demo failed:", error);
  }

  function tick(timestamp) {
    frame = null;
    try {
      if (previous !== null) remainder = core.advance(state, model, (timestamp - previous) / 1000, remainder);
      previous = timestamp;
      render();
      frame = requestAnimationFrame(tick);
    } catch (error) {
      fail(error);
    }
  }

  function synchronize() {
    if (!model || failed) return;
    const running = !paused && visible && !document.hidden;
    button.textContent = paused ? "Resume demo" : "Pause demo";
    status.textContent = paused ? "Paused" : running ? "Live model" : "Suspended";
    if (running && frame === null) {
      previous = null;
      frame = requestAnimationFrame(tick);
    } else if (!running) {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      previous = null;
      remainder = 0;
    }
  }

  button.addEventListener("click", () => {
    paused = !paused;
    synchronize();
  });
  motion.addEventListener("change", event => {
    if (event.matches) paused = true;
    synchronize();
  });
  document.addEventListener("visibilitychange", synchronize);
  const observer = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    synchronize();
  });
  observer.observe(demo);

  async function load() {
    try {
      const response = await fetch(new URL("model.json", document.currentScript.src));
      if (!response.ok) throw new Error(`Model request failed: HTTP ${response.status}`);
      model = core.validateModel(await response.json());
      state = core.createState();
      render();
      button.disabled = false;
      synchronize();
    } catch (error) {
      fail(error);
    }
  }
  load();
})();
