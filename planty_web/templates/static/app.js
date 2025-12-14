async function fetchState() {
  const r = await fetch("/api/v1/state");
  const s = await r.json();

  const sensorEl = document.getElementById("sensor");
  const visionEl = document.getElementById("vision");
  const lightEl = document.getElementById("light_state");

  if (!s.sensor) {
    sensorEl.innerHTML = "<i>No sensor data yet</i>";
  } else {
    sensorEl.innerHTML = `
      <div>Temp: <b>${s.sensor.temperature}</b> °C</div>
      <div>Humidity: <b>${s.sensor.humidity}</b> %</div>
      <div>Water level: <b>${s.sensor.water_level_cm}</b> cm</div>
      <div>ts: ${new Date(s.sensor.ts).toLocaleString()}</div>
    `;
  }

  if (!s.vision) {
    visionEl.innerHTML = "<i>No vision data yet</i>";
  } else {
    const isHealthy = (s.vision.health_status === "healthy");
    const tag = isHealthy ? "ok" : "bad";
    visionEl.innerHTML = `
      <div>Status: <span class="${tag}">${s.vision.health_status.toUpperCase()}</span></div>
      <div>Health score: <b>${s.vision.health_score}</b></div>
      <div>Confidence: <b>${(s.vision.confidence ?? 0).toFixed(3)}</b></div>
      <div>Raw label: <code>${s.vision.raw_label ?? ""}</code></div>
      <div>ts: ${new Date(s.vision.ts).toLocaleString()}</div>
    `;
  }

  if (!s.light_state) {
    lightEl.innerHTML = "<i>unknown</i>";
  } else {
    lightEl.innerHTML = `State: <b>${s.light_state.state.toUpperCase()}</b> (ts: ${new Date(s.light_state.ts).toLocaleTimeString()})`;
  }
}

async function postLight(state) {
  const msg = document.getElementById("light_msg");
  msg.textContent = "Sending...";
  try {
    const r = await fetch("/api/v1/light", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({state})
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "RPC failed");
    msg.innerHTML = `<span class="ok">OK</span> light = ${data.state}`;
  } catch (e) {
    msg.innerHTML = `<span class="bad">FAIL</span> ${e.message}`;
  }
}

function lightOn(){ postLight("on"); }
function lightOff(){ postLight("off"); }

setInterval(fetchState, 1000);
fetchState();
