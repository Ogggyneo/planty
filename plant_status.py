import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# =========================
# 1. CẤU HÌNH
# =========================

WEIGHTS_PATH = "tomato_disease_model_weights.pth"
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

HEALTHY_CLASS_NAME = "Tomato___healthy"  # <-- dùng đúng tên trong CLASS_NAMES


# =========================
# 2. HÀM TIỆN ÍCH DEVICE
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
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)

        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

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
        raise FileNotFoundError(f"Không tìm thấy file: {weights_path}")

    device = get_default_device()
    print("✅ Dùng device:", device)

    num_classes = len(CLASS_NAMES)
    model = CNN_NeuralNet(in_channels=3, num_diseases=num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = to_device(model, device)
    model.eval()
    print(f"✅ Đã load model từ {weights_path}")
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
    img_t = img_t.unsqueeze(0)
    return img_t


# =========================
# 6. PREDICT (UPDATED: 2 TRẠNG THÁI)
# =========================

def predict_frame_binary(frame_bgr, model, device):
    """
    Trả về:
      - health_status: 'healthy' hoặc 'unhealthy'
      - conf: độ tự tin của class dự đoán top-1 (0..1)
      - raw_label: label gốc (để debug)
    """
    xb = preprocess_frame(frame_bgr)
    xb = to_device(xb, device)

    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        raw_label = CLASS_NAMES[idx]
        conf = probs[0, idx].item()

    health_status = "healthy" if raw_label == HEALTHY_CLASS_NAME else "unhealthy"
    return health_status, conf, raw_label


# =========================
# 7. WEBCAM DEMO
# =========================

def run_webcam_demo():
    model, device = load_trained_model(WEIGHTS_PATH)

    cap = cv2.VideoCapture(1)  # nếu không được thử 0
    if not cap.isOpened():
        print("❌ Không mở được webcam. Thử đổi index (0/1/2).")
        return

    print("🎥 Webcam đang chạy. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Không đọc được frame từ camera.")
            break

        health_status, conf, raw_label = predict_frame_binary(frame, model, device)

        # text cho demo: chỉ hiện 2 trạng thái + confidence
        text = f"{health_status.upper()} ({conf*100:.1f}%)"

        # màu: healthy xanh, unhealthy đỏ
        color = (0, 255, 0) if health_status == "healthy" else (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        # (optional) debug nhỏ ở dòng dưới: label gốc
        cv2.putText(
            frame,
            f"raw: {raw_label}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Plant Health (Binary) - PyTorch Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
