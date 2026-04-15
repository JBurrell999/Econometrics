from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

CLASSES = ["Cat", "Dog"]
IMG_SIZE = 224
OUT_CSV = Path("predictions.csv")
EXPECTED_COLUMNS = ["filename", "pred_label", "p_cat", "p_dog"]
IMAGE_DIR_CANDIDATES = [Path("Challenge_Images"), Path("challenge_images")]
WEIGHTS_PATH = Path("/Users/jjburrell/Downloads/efficientnetv2_m_best.pt")


def resolve_image_dir() -> Path | None:
    for candidate in IMAGE_DIR_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


device = "cpu"
model = timm.create_model(
    "tf_efficientnetv2_m.in21k_ft_in1k", pretrained=False, num_classes=2
)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.to(device).eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

EXTS = {".jpg", ".jpeg", ".png"}
image_dir = resolve_image_dir()
paths = (
    sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in EXTS)
    if image_dir is not None
    else []
)


def classify(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        p_cat, p_dog = F.softmax(model(x), dim=1)[0].tolist()
    return {
        "filename": path.name,
        "pred_label": CLASSES[0 if p_cat > p_dog else 1],
        "p_cat": p_cat,
        "p_dog": p_dog,
    }


if not paths:
    searched = ", ".join(str(path) for path in IMAGE_DIR_CANDIDATES)
    print(f"No challenge images found. Looked in: {searched}")

df = pd.DataFrame((classify(p) for p in tqdm(paths, desc="classifying")), columns=EXPECTED_COLUMNS)
df.to_csv(OUT_CSV, index=False)
print(df["pred_label"].value_counts())
print(df.head())
