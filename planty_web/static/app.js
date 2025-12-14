function fmt(item) {
  if (!item) return "<i>no data</i>";
  const ts = item.ts ? new Date(item.ts).toLocaleString() : "n/a";
  return `${item.value} <span style="color:#888">(ts: ${ts})</span>`;
}

async function refresh() {
  const el = document.getElementById("telemetry");
  const r = await fetch("/api/state");
  const j = await r.json();
  if (!j.ok) {
    el.innerHTML = `<span class="bad">ERROR</span> ${j.error}`;
    return;
  }
  const d = j.data;
  el.innerHTML = `
    <div>temperature: <b>${fmt(d.temperature)}</b></div>
    <div>humidity: <b>${fmt(d.humidity)}</b></div>
    <div>water_low: <b>${fmt(d.water_low)}</b></div>
    <div>pump_on: <b>${fmt(d.pump_on)}</b></div>
    <div>manual_mode: <b>${fmt(d.manual_mode)}</b></div>
  `;
}

async function post(url, body) {
  const msg = document.getElementById("msg");
  msg.textContent = "Sending...";
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "request failed");
    msg.innerHTML = `<span class="ok">OK</span> ${JSON.stringify(j)}`;
  } catch (e) {
    msg.innerHTML = `<span class="bad">FAIL</span> ${e.message}`;
  }
}

function setManual(enabled) { post("/api/manual", {enabled}); }
function setPump(state) { post("/api/pump", {state}); }

setInterval(refresh, 1000);
refresh();
