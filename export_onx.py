import os
import torch
import torch.nn as nn

WEIGHTS_PATH = "tomato_disease_model_weights.pth"
ONNX_OUT = "tomato_model.onnx"
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

if __name__ == "__main__":
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(WEIGHTS_PATH)

    model = CNN_NeuralNet(3, len(CLASS_NAMES))
    sd = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)  # NCHW, float32
    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
    )
    print("✅ Exported:", ONNX_OUT)
