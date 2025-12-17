import os
import time
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

TB_BASE_URL = "https://thingsboard.cloud"
TB_USERNAME = "01.dhn.work@gmail.com"  # email đăng nhập ThingsBoard
TB_PASSWORD = "dinhhongngoc"  # password
TB_DEVICE_ID = "e3e03670-d8ce-11f0-afc5-358ed627911d" # deviceId (UUID) của YOLOUNO1
print("TB_DEVICE_ID=", TB_USERNAME)



# keys bạn đang publish
TELEMETRY_KEYS = ["temperature", "humidity", "water_low", "pump_on", "manual_mode"]

# JWT cache
_jwt = None
_jwt_exp = 0.0


def tb_login() -> str:
    global _jwt, _jwt_exp
    if not TB_USERNAME or not TB_PASSWORD:
        raise RuntimeError("Missing TB_USERNAME/TB_PASSWORD")

    if _jwt and time.time() < _jwt_exp:
        return _jwt

    r = requests.post(
        f"{TB_BASE_URL}/api/auth/login",
        json={"username": TB_USERNAME, "password": TB_PASSWORD},
        timeout=10,
    )
    if r.status_code != 200:
        raise RuntimeError(f"TB login failed: {r.status_code} {r.text}")

    data = r.json()
    token = data.get("token") or data.get("jwtToken") or data.get("accessToken")
    if not token:
        raise RuntimeError(f"TB login missing token: {data}")

    _jwt = token
    _jwt_exp = time.time() + 600
    return _jwt


def tb_headers() -> dict:
    jwt = tb_login()
    return {"X-Authorization": f"Bearer {jwt}"}


def tb_get_latest_telemetry(device_id: str) -> dict:
    if not device_id:
        raise RuntimeError("Missing TB_DEVICE_ID")
    keys = ",".join(TELEMETRY_KEYS)
    url = f"{TB_BASE_URL}/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
    r = requests.get(url, headers=tb_headers(), params={"keys": keys}, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"TB get telemetry failed: {r.status_code} {r.text}")

    # TB trả dạng: { "temperature":[{"ts":..., "value":"27.1"}], ... }
    raw = r.json()
    out = {}
    for k in TELEMETRY_KEYS:
        arr = raw.get(k, [])
        if arr:
            out[k] = {"ts": arr[0].get("ts"), "value": arr[0].get("value")}
        else:
            out[k] = None
    return out


def tb_rpc_oneway(device_id: str, method: str, params):
    url = f"{TB_BASE_URL}/api/plugins/rpc/oneway/{device_id}"
    payload = {"method": method, "params": params}
    r = requests.post(url, headers=tb_headers(), json=payload, timeout=10)
    if r.status_code not in (200, 202):
        raise RuntimeError(f"TB RPC failed: {r.status_code} {r.text}")


app = FastAPI(title="Planty Webserver", version="1.0.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class PumpCmd(BaseModel):
    state: bool  # true=ON, false=OFF


class ManualCmd(BaseModel):
    enabled: bool


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/state")
def api_state():
    try:
        data = tb_get_latest_telemetry(TB_DEVICE_ID)
        return JSONResponse({"ok": True, "data": data, "server_time_ms": int(time.time()*1000)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/manual")
def api_manual(cmd: ManualCmd):
    try:
        tb_rpc_oneway(TB_DEVICE_ID, "setManualMode", cmd.enabled)
        return {"ok": True, "manual": cmd.enabled}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/pump")
def api_pump(cmd: PumpCmd):
    try:
        tb_rpc_oneway(TB_DEVICE_ID, "setPump", cmd.state)
        return {"ok": True, "pump": cmd.state}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
