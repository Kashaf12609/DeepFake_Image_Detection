
import io
import os

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "best_model.pth"
)

DEVICE = torch.device("cpu")
THRESHOLD = 0.15

model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, 1
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def home():
    return {"message": "Deepfake Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg", "image/png", "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Upload JPG, PNG or WEBP image."
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be under 10 MB."
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        tensor = preprocess(image).unsqueeze(0)

        with torch.inference_mode():
            output = model(tensor)
            probability = torch.sigmoid(output).item()

        prediction = (
            "FAKE" if probability >= THRESHOLD else "REAL"
        )

        return {
            "prediction": prediction,
            "fake_probability": round(probability, 4),
            "threshold": THRESHOLD
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not process image."
        )
