"""
tomato_cam_pytorch.py

Chạy real-time webcam detect bệnh lá cà chua
dùng PyTorch + model CNN_NeuralNet (state_dict trong tomato_disease_model_weights.pth)

Yêu cầu:
- pip install torch torchvision opencv-python pillow
- Đặt file tomato_disease_model_weights.pth cùng thư mục với script này
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# =========================
# 1. CẤU HÌNH
# =========================

# 👉 SỬA LẠI NẾU TÊN FILE KHÁC
WEIGHTS_PATH = "tomato_disease_model_weights.pth"

IMG_SIZE = 256  # model được train với ảnh 256x256

# 10 class bệnh lá cà chua (thứ tự alphabet, giống ImageFolder)
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


# =========================
# 2. HÀM TIỆN ÍCH DEVICE
# =========================

def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(x, device):
    return x.to(device, non_blocking=True)


# =========================
# 3. ĐỊNH NGHĨA MODEL (Y HỆT LÚC TRAIN)
# =========================

def ConvBlock(in_channels, out_channels, pool=False):
    """
    Conv2d + BatchNorm + ReLU (+ MaxPool2d(4) nếu pool=True)
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class CNN_NeuralNet(nn.Module):
    """
    Kiến trúc:

    conv1 -> conv2 (pool) -> res1 (2 conv block) + skip ->
    conv3 (pool) -> conv4 (pool) -> res2 (2 conv block) + skip ->
    MaxPool(4) -> Flatten -> Linear(512 -> num_classes)
    """

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
# 4. LOAD STATE_DICT TỪ .PTH
# =========================

def load_trained_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file: {weights_path}")

    device = get_default_device()
    print("✅ Dùng device:", device)

    num_classes = len(CLASS_NAMES)
    model = CNN_NeuralNet(in_channels=3, num_diseases=num_classes)

    # load state_dict
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = to_device(model, device)
    model.eval()
    print(f"✅ Đã load model từ {weights_path}")
    return model, device


# =========================
# 5. TIỀN XỬ LÝ FRAME TỪ WEBCAM
# =========================

# Resize + ToTensor (giống lúc train: chỉ ToTensor với ảnh 256x256)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # scale về [0,1]
])


def preprocess_frame(frame_bgr):
    """
    frame_bgr: numpy array (H,W,3) đọc từ OpenCV
    return: Tensor [1,3,256,256]
    """
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = transform(frame_rgb)          # [3,256,256]
    img_t = img_t.unsqueeze(0)            # [1,3,256,256]
    return img_t


def predict_frame(frame_bgr, model, device):
    """
    Trả về (label, idx) dự đoán cho 1 frame BGR
    """
    xb = preprocess_frame(frame_bgr)
    xb = to_device(xb, device)

    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        label = CLASS_NAMES[idx]
        conf = probs[0, idx].item()

    return label, conf


# =========================
# 6. CHẠY WEBCAM
# =========================

def run_webcam_demo():
    model, device = load_trained_model(WEIGHTS_PATH)

    cap = cv2.VideoCapture(1)  # nếu không được thử 1,2...
    if not cap.isOpened():
        print("❌ Không mở được webcam, kiểm tra camera/index.")
        return

    print("🎥 Webcam đang chạy. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Không đọc được frame từ camera.")
            break

        label, conf = predict_frame(frame, model, device)

        text = f"{label} ({conf*100:.1f}%)"

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Tomato Leaf Disease - PyTorch Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# 7. MAIN
# =========================

if __name__ == "__main__":
    run_webcam_demo()
