(function (root) {
  "use strict";

  const WIDTH = 476;
  const HEIGHT = 500;
  const SPEED = 2.5;
  const STEP_SECONDS = 1 / 144;
  const architecture = [7, 8, 6, 2];
  const activations = ["tanh", "relu", "sigmoid"];
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function validateModel(model) {
    if (!model || model.version !== 1 ||
        JSON.stringify(model.architecture) !== JSON.stringify(architecture) ||
        !/^[a-f0-9]{64}$/.test(model.checkpoint_sha256) ||
        !Array.isArray(model.layers) || model.layers.length !== 3) {
      throw new Error("Unsupported Pong model metadata.");
    }
    model.layers.forEach((layer, index) => {
      const inputs = architecture[index];
      const outputs = architecture[index + 1];
      if (!layer || layer.activation !== activations[index] ||
          !Array.isArray(layer.weight) || layer.weight.length !== outputs ||
          !layer.weight.every(row => Array.isArray(row) && row.length === inputs &&
            row.every(Number.isFinite)) ||
          !Array.isArray(layer.bias) || layer.bias.length !== outputs ||
          !layer.bias.every(Number.isFinite)) {
        throw new Error(`Invalid Pong model layer ${index}.`);
      }
    });
    return model;
  }

  function predict(model, observations) {
    if (observations.length !== 7 || !observations.every(Number.isFinite)) {
      throw new Error("Pong inference requires seven finite observations.");
    }
    let values = observations;
    for (const layer of model.layers) {
      values = layer.weight.map((row, index) => {
        const sum = row.reduce((total, weight, input) => total + weight * values[input], layer.bias[index]);
        if (layer.activation === "tanh") return Math.tanh(sum);
        if (layer.activation === "relu") return Math.max(0, sum);
        return 1 / (1 + Math.exp(-sum));
      });
    }
    return {raw: values, actions: values.map(value => value > 0.5)};
  }

  function observe(state) {
    const {ball, ai} = state;
    const magnitude = Math.hypot(ball.vx, ball.vy);
    return [
      (ball.x - ai.x) / WIDTH,
      (ball.y - ai.y) / HEIGHT,
      (ai.x * 2 - WIDTH) / WIDTH,
      (ball.x * 2 - WIDTH) / WIDTH,
      (ball.y * 2 - HEIGHT) / HEIGHT,
      ball.vx / magnitude,
      ball.vy / magnitude,
    ];
  }

  function serve(state, random) {
    const integer = (low, high) => low + Math.floor(random() * (high - low + 1));
    const sign = () => random() < 0.5 ? -1 : 1;
    const ball = state.ball;
    ball.x = integer(10, WIDTH - 10);
    ball.y = integer(HEIGHT / 2 - 100, HEIGHT / 2 + 100);
    const verticalSign = sign();
    ball.vx = (0.3 + 0.5 * random()) * sign() * 5 * 0.4;
    ball.vy = verticalSign * (5 - Math.abs(ball.vx));
  }

  function createState(random = Math.random) {
    const state = {
      ball: {x: 0, y: 0, vx: 0, vy: 0, width: 12, height: 12},
      ai: {x: WIDTH / 2, y: HEIGHT - 5, width: 80, height: 10},
      cpu: {x: WIDTH / 2, y: 5, width: 80, height: 10},
      scores: {ai: 0, cpu: 0},
    };
    serve(state, random);
    return state;
  }

  function moveAI(state, actions) {
    if (actions[0] === actions[1]) return;
    state.ai.x = clamp(state.ai.x + (actions[0] ? -1 : 1) * Math.max(2, SPEED), 40, WIDTH - 40);
  }

  function integrate(ball) {
    ball.x += ball.vx;
    ball.y += ball.vy;
  }

  function collides(ball, paddle) {
    return Math.abs(ball.x - paddle.x) <= (ball.width + paddle.width) / 2 &&
      Math.abs(ball.y - paddle.y) <= (ball.height + paddle.height) / 2;
  }

  function bounce(state, paddle, isCPU) {
    const ball = state.ball;
    ball.vy *= -1;
    const angle = Math.acos(clamp(ball.vx / Math.hypot(ball.vx, ball.vy), -1, 1)) * 180 / Math.PI;
    // Python uses floor division, not truncation, for the hit-position influence.
    const influence = Math.floor((ball.x - paddle.x) / (paddle.width / 2));
    const rotation = angle > 15 ? influence * 30 * (isCPU ? -1 : 1) : 0;
    const radians = rotation * Math.PI / 180;
    const {vx, vy} = ball;
    ball.vx = vx * Math.cos(radians) - vy * Math.sin(radians);
    ball.vy = vx * Math.sin(radians) + vy * Math.cos(radians);
    const newAngle = Math.acos(clamp(ball.vx / Math.hypot(ball.vx, ball.vy), -1, 1));
    if (newAngle === 0) ball.vy = 0.2;
    integrate(ball);
  }

  function step(state, model, random = Math.random) {
    moveAI(state, predict(model, observe(state)).actions);
    const {ball, cpu} = state;
    if (ball.vy < 0) {
      cpu.x = clamp(cpu.x + clamp((ball.x - cpu.x) * 0.1, -SPEED, SPEED), 40, WIDTH - 40);
    }
    // Match PlayZone's collision-before-integration order, including its nudges.
    if (ball.x + 6 > WIDTH || ball.x - 6 < 0) {
      ball.vx *= -1;
      integrate(ball);
    }
    if (ball.y + 6 > HEIGHT) {
      state.scores.cpu += 1;
      serve(state, random);
    } else if (ball.y - 6 < 0) {
      state.scores.ai += 1;
      serve(state, random);
    }
    if (collides(ball, state.ai)) bounce(state, state.ai, false);
    if (collides(ball, cpu)) bounce(state, cpu, true);
    integrate(ball);
  }

  function displayPosition(entity) {
    // Rotate top/bottom training coordinates into left/right display coordinates.
    return {left: entity.y / HEIGHT * 100, top: (1 - entity.x / WIDTH) * 100};
  }

  function advance(state, model, elapsed, remainder = 0, random = Math.random) {
    let accumulated = remainder + clamp(elapsed, 0, 0.1);
    while (accumulated >= STEP_SECONDS) {
      step(state, model, random);
      accumulated -= STEP_SECONDS;
    }
    return accumulated;
  }

  root.PongCore = {WIDTH, HEIGHT, STEP_SECONDS, validateModel, predict, observe,
    createState, moveAI, step, displayPosition, advance};
})(globalThis);
