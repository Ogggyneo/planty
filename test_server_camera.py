import os
import time
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import ssl

# =========================
# 0. MQTT (ThingsBoard) CONFIG
# =========================
# Cách dùng nhanh:
#   set TB_HOST=demo.thingsboard.io
#   set TB_DEVICE_TOKEN=YOUR_TOKEN
# TB_HOST = os.getenv("TB_HOST", "demo.thingsboard.io")
# TB_PORT = int(os.getenv("TB_PORT", "1883"))
TB_DEVICE_TOKEN = os.getenv("TB_DEVICE_TOKEN", "PUT_YOUR_DEVICE_TOKEN_HERE")

TB_HOST = os.getenv("TB_HOST", "thingsboard.cloud")
TB_PORT = int(os.getenv("TB_PORT", "8883"))
TB_DEVICE_TOKEN = os.getenv("TB_DEVICE_TOKEN", "PUT_YOUR_DEVICE_TOKEN_HERE")

PUBLISH_EVERY_SEC = float(os.getenv("PUBLISH_EVERY_SEC", "2.0"))
CLIENT_ID = os.getenv("TB_CLIENT_ID", "rasp-vision-binary")
# =========================
# 1. MODEL CONFIG
# =========================
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "tomato_disease_model_weights.pth")
IMG_SIZE = 256

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
HEALTHY_CLASS_NAME = "Tomato___healthy"

# =========================
# 2. DEVICE UTILS
# =========================
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_device(x, device):
    return x.to(device, non_blocking=True)

# =========================
# 3. MODEL
# =========================
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# =========================
# 4. LOAD MODEL
# =========================
def load_trained_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file weights: {weights_path}")

    device = get_default_device()
    print("✅ Device:", device)

    model = CNN_NeuralNet(in_channels=3, num_diseases=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = to_device(model, device)
    model.eval()
    print(f"✅ Loaded weights: {weights_path}")
    return model, device

# =========================
# 5. PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = transform(frame_rgb)
    return img_t.unsqueeze(0)

# =========================
# 6. PREDICT (BINARY)
# =========================
def predict_frame_binary(frame_bgr, model, device):
    xb = preprocess_frame(frame_bgr)
    xb = to_device(xb, device)

    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        raw_label = CLASS_NAMES[idx]
        conf = float(probs[0, idx].item())

    health_status = "healthy" if raw_label == HEALTHY_CLASS_NAME else "unhealthy"

    # health_score 0-100 (đơn giản, ổn demo)
    if health_status == "healthy":
        health_score = int(round(70 + 30 * conf))          # 70..100
    else:
        health_score = int(round(60 - 60 * conf))          # 0..60 (bệnh càng chắc -> càng thấp)
        health_score = max(0, min(60, health_score))

    return health_status, health_score, conf, raw_label

# =========================
# 7. MQTT PUBLISHER (ThingsBoard)
# =========================
def create_tb_mqtt_client():
    try:
        import paho.mqtt.client as mqtt
    except ImportError as e:
        raise SystemExit(
            "Thiếu paho-mqtt. Cài bằng: pip install paho-mqtt"
        ) from e

    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.username_pw_set(TB_DEVICE_TOKEN)  # ThingsBoard auth = token as username
    # bật TLS cho cloud
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.tls_insecure_set(False)

    client.connect(TB_HOST, TB_PORT, keepalive=60)
    client.loop_start()
    return client

def tb_publish_telemetry(client, data: dict):
    payload = json.dumps(data, ensure_ascii=False)
    # ThingsBoard telemetry topic:
    # v1/devices/me/telemetry
    result = client.publish("v1/devices/me/telemetry", payload=payload, qos=0)
    return result.rc == 0

# =========================
# 8. WEBCAM LOOP + THROTTLE PUBLISH
# =========================
def run_webcam_binary_and_publish():
    if TB_DEVICE_TOKEN == "PUT_YOUR_DEVICE_TOKEN_HERE":
        print("⚠️ Bạn chưa set TB_DEVICE_TOKEN. Set env var hoặc sửa trực tiếp trong code.")
        print("   Ví dụ (Windows PowerShell):")
        print('   $env:TB_HOST="demo.thingsboard.io"; $env:TB_DEVICE_TOKEN="XXXX"')
        # vẫn cho chạy webcam để debug, nhưng sẽ không publish
        publish_enabled = False
        mqtt_client = None
    else:
        publish_enabled = True
        mqtt_client = create_tb_mqtt_client()
        print(f"✅ MQTT connected: {TB_HOST}:{TB_PORT}")

    model, device = load_trained_model(WEIGHTS_PATH)

    # camera index: thử 0 trước, fallback 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Không mở được camera (thử index 0/1).")
        return

    print("🎥 Running... press 'q' to quit.")
    last_pub = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không đọc được frame.")
            break

        health_status, health_score, conf, raw_label = predict_frame_binary(frame, model, device)

        # overlay
        color = (0, 255, 0) if health_status == "healthy" else (0, 0, 255)
        cv2.putText(frame, f"{health_status.upper()} | score={health_score} | conf={conf*100:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"raw: {raw_label}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # publish throttle
        now = time.time()
        if publish_enabled and (now - last_pub) >= PUBLISH_EVERY_SEC:
            telemetry = {
                "health_status": health_status,
                "health_score": health_score,
                "confidence": round(conf, 4),
                "raw_label": raw_label,
            }
            ok = tb_publish_telemetry(mqtt_client, telemetry)
            if not ok:
                print("⚠️ Publish failed")
            last_pub = now

        cv2.imshow("Plant Health (Binary) + ThingsBoard Telemetry", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if publish_enabled and mqtt_client is not None:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    run_webcam_binary_and_publish()
