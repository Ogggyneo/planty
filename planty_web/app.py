import os
import time
import sqlite3
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# =========================================================
# CONFIG
# =========================================================
DB_PATH = os.getenv("DB_PATH", "planty.db")

# API key để devices post telemetry trực tiếp về webserver này
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "change-me")

# ThingsBoard Cloud (để gọi RPC xuống actuator)
TB_BASE_URL = os.getenv("TB_BASE_URL", "https://thingsboard.cloud")
TB_USERNAME = os.getenv("TB_USERNAME", "")      # email đăng nhập ThingsBoard tenant
TB_PASSWORD = os.getenv("TB_PASSWORD", "")      # password
ACTUATOR_DEVICE_ID = os.getenv("ACTUATOR_DEVICE_ID", "")  # deviceId của actuator trên TB

# RPC method names (bạn có thể đổi theo firmware)
RPC_LIGHT_ON_METHOD = os.getenv("RPC_LIGHT_ON_METHOD", "Light_ON")
RPC_LIGHT_OFF_METHOD = os.getenv("RPC_LIGHT_OFF_METHOD", "Light_OFF")

# =========================================================
# DB
# =========================================================
def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

conn = db_connect()
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS telemetry_sensor (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  temperature REAL,
  humidity REAL,
  water_level_cm REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS telemetry_vision (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  health_status TEXT,
  health_score INTEGER,
  confidence REAL,
  raw_label TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS actuator_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  action TEXT,
  ok INTEGER,
  detail TEXT
)
""")

conn.commit()

# cache latest (để dashboard nhẹ)
LATEST: Dict[str, Any] = {
    "sensor": None,
    "vision": None,
    "light_state": None,
}

# =========================================================
# MODELS
# =========================================================
class SensorPayload(BaseModel):
    temperature: float = Field(..., description="Celsius")
    humidity: float = Field(..., description="Percent")
    water_level_cm: float = Field(..., description="cm")

class VisionPayload(BaseModel):
    health_status: str = Field(..., description="healthy|unhealthy")
    health_score: int = Field(..., ge=0, le=100)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    raw_label: Optional[str] = None

class LightCommand(BaseModel):
    state: str = Field(..., description="on|off")

# =========================================================
# THINGSBOARD REST HELPERS
# =========================================================
_tb_jwt: Optional[str] = None
_tb_jwt_exp: float = 0.0

def tb_login_get_jwt() -> str:
    """
    ThingsBoard login -> JWT token.
    Endpoint thường là /api/auth/login (xem Swagger UI trên instance).
    """
    global _tb_jwt, _tb_jwt_exp

    if not TB_USERNAME or not TB_PASSWORD:
        raise RuntimeError("Missing TB_USERNAME/TB_PASSWORD env vars")

    # reuse token 10 phút
    if _tb_jwt and time.time() < _tb_jwt_exp:
        return _tb_jwt

    url = f"{TB_BASE_URL}/api/auth/login"
    r = requests.post(
        url,
        json={"username": TB_USERNAME, "password": TB_PASSWORD},
        timeout=10,
    )
    if r.status_code != 200:
        raise RuntimeError(f"TB login failed: {r.status_code} {r.text}")

    data = r.json()
    token = data.get("token") or data.get("jwtToken") or data.get("accessToken")
    if not token:
        raise RuntimeError(f"TB login response missing token: {data}")

    _tb_jwt = token
    _tb_jwt_exp = time.time() + 600
    return token

def tb_rpc_oneway(device_id: str, method: str, params: dict) -> bool:
    """
    Server-side RPC one-way.
    Endpoint phổ biến: /api/plugins/rpc/oneway/{deviceId}
    (Bạn có thể kiểm tra lại ở Swagger UI của ThingsBoard instance.)
    """
    jwt = tb_login_get_jwt()
    url = f"{TB_BASE_URL}/api/plugins/rpc/oneway/{device_id}"
    headers = {"X-Authorization": f"Bearer {jwt}"}
    payload = {"method": method, "params": params}

    r = requests.post(url, headers=headers, json=payload, timeout=10)
    return r.status_code in (200, 202)

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(title="Planty Webserver", version="1.0.0")
templates = Jinja2Templates(directory="templates")

def require_ingest_key(x_api_key: Optional[str]):
    if x_api_key != INGEST_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/v1/state")
def get_state():
    return JSONResponse({
        "sensor": LATEST["sensor"],
        "vision": LATEST["vision"],
        "light_state": LATEST["light_state"],
        "server_time_ms": int(time.time() * 1000),
    })

@app.post("/api/v1/telemetry/sensor")
def ingest_sensor(payload: SensorPayload, x_api_key: Optional[str] = Header(None)):
    require_ingest_key(x_api_key)
    ts = int(time.time() * 1000)
    cur.execute(
        "INSERT INTO telemetry_sensor(ts, temperature, humidity, water_level_cm) VALUES (?,?,?,?)",
        (ts, payload.temperature, payload.humidity, payload.water_level_cm),
    )
    conn.commit()
    LATEST["sensor"] = {"ts": ts, **payload.model_dump()}
    return {"ok": True}

@app.post("/api/v1/telemetry/vision")
def ingest_vision(payload: VisionPayload, x_api_key: Optional[str] = Header(None)):
    require_ingest_key(x_api_key)

    hs = payload.health_status.lower().strip()
    if hs not in ("healthy", "unhealthy"):
        raise HTTPException(status_code=422, detail="health_status must be healthy|unhealthy")

    ts = int(time.time() * 1000)
    cur.execute(
        "INSERT INTO telemetry_vision(ts, health_status, health_score, confidence, raw_label) VALUES (?,?,?,?,?)",
        (ts, hs, payload.health_score, payload.confidence, payload.raw_label),
    )
    conn.commit()
    LATEST["vision"] = {"ts": ts, **payload.model_dump()}
    return {"ok": True}

@app.post("/api/v1/light")
def set_light(cmd: LightCommand):
    """
    User bấm trên web -> webserver gọi ThingsBoard RPC -> actuator bật/tắt đèn.
    """
    if not ACTUATOR_DEVICE_ID:
        raise HTTPException(status_code=500, detail="Missing ACTUATOR_DEVICE_ID")

    s = cmd.state.lower().strip()
    if s not in ("on", "off"):
        raise HTTPException(status_code=422, detail="state must be on|off")

    # gọi RPC
    if s == "on":
        ok = tb_rpc_oneway(ACTUATOR_DEVICE_ID, RPC_LIGHT_ON_METHOD, {})
    else:
        ok = tb_rpc_oneway(ACTUATOR_DEVICE_ID, RPC_LIGHT_OFF_METHOD, {})

    ts = int(time.time() * 1000)
    cur.execute(
        "INSERT INTO actuator_log(ts, action, ok, detail) VALUES (?,?,?,?)",
        (ts, f"light_{s}", 1 if ok else 0, ""),
    )
    conn.commit()

    if ok:
        LATEST["light_state"] = {"ts": ts, "state": s}
        return {"ok": True, "state": s}
    raise HTTPException(status_code=502, detail="RPC failed (check TB credentials/device online/RPC method name)")
