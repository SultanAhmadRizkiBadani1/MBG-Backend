# =========================================================
# MBG AI BACKEND
# FASTAPI + YOLOv8 + NUTRITION ANALYSIS
# =========================================================

# =========================================================
# IMPORT LIBRARY
# =========================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

from PIL import Image

import pandas as pd
import numpy as np

import shutil
import os
import uuid


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(

    title="MBG AI Backend",

    version="1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# FOLDER
# =========================================================

UPLOAD_FOLDER = "uploads"

RESULT_FOLDER = "results"


os.makedirs(

    UPLOAD_FOLDER,

    exist_ok=True
)

os.makedirs(

    RESULT_FOLDER,

    exist_ok=True
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO(

    "best.pt"
)

print("\n✅ YOLO MODEL LOADED")


# =========================================================
# LOAD NUTRITION CSV
# =========================================================

nutrition_df = pd.read_csv(

    "nutrition.csv"
)

print("\n✅ NUTRITION CSV LOADED")


# =========================================================
# FOOD MAPPING
# =========================================================

FOOD_MAPPING = {

    # AYAM
    "ayam": "Ayam",
    "ayam goreng": "Ayam goreng",
    "ayam teriyaki": "Chicken teriyaki masakan",

    # NASI
    "nasi": "Nasi",
    "nasi putih": "Nasi",
    "nasi kuning": "Nasi",
    "nasi goreng": "Nasi Goreng",

    # TEMPE
    "tempe": "Tempe",

    # JERUK
    "jeruk": "Jeruk Manis",

    # TIMUN
    "timun": "Ketimun",
    "mentimun": "Ketimun",

    # SUSU
    "susu": "Susu Kedelai",

    # BUAH
    "pisang": "Pisang",
    "apel": "Apel"
}


# =========================================================
# GET NUTRITION FUNCTION
# =========================================================

def get_nutrition(food_name):

    food_name = food_name.lower().strip()

    mapped_name = FOOD_MAPPING.get(
        food_name,
        food_name
    )

    result = nutrition_df[

        nutrition_df["name"]
        .astype(str)
        .str.lower()
        .str.contains(
            mapped_name.lower(),
            na=False
        )

    ]

    if not result.empty:

        row = result.iloc[0]

        return {

            "kalori":
            float(row["calories"]),

            "protein":
            float(row["proteins"]),

            "karbo":
            float(row["carbohydrate"]),

            "lemak":
            float(row["fat"])

        }

    print(
        f"Nutrisi tidak ditemukan: {food_name}"
    )

    return {

        "kalori":0,
        "protein":0,
        "karbo":0,
        "lemak":0

    }


# =========================================================
# HOME API
# =========================================================

@app.get("/")

def home():

    return {

        "message":

        "🚀 MBG AI Backend Running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")

def health():

    return {

        "status": "healthy",

        "model": "YOLOv8"
    }


# =========================================================
# PREDICT API
# =========================================================

@app.post("/predict")

async def predict(

    file: UploadFile = File(...)
):


    # =====================================
    # VALIDASI EXTENSION
    # =====================================

    allowed_extensions = [

        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]


    ext = os.path.splitext(

        file.filename

    )[1].lower()


    if ext not in allowed_extensions:

        return {

            "success": False,

            "error":

            "Format gambar harus JPG PNG WEBP"
        }


    # =====================================
    # CREATE UNIQUE FILE
    # =====================================

    file_id = str(uuid.uuid4())

    file_name = f"{file_id}{ext}"

    file_path = os.path.join(

        UPLOAD_FOLDER,

        file_name
    )


    # =====================================
    # SAVE FILE
    # =====================================

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(

            file.file,

            buffer
        )


    # =====================================
    # LOAD IMAGE PIL
    # =====================================

    try:

        image = Image.open(

            file_path

        ).convert("RGB")


    except Exception as e:

        return {

            "success": False,

            "error":

            f"Gagal membaca gambar: {str(e)}"
        }


    # =====================================
    # CONVERT NUMPY
    # =====================================

    image_np = np.array(image)


    # =====================================
    # VALIDASI IMAGE
    # =====================================

    if image_np is None:

        return {

            "success": False,

            "error":

            "Image tidak valid"
        }


    # =====================================
    # YOLO PREDICT
    # =====================================

    try:

        results = model.predict(
            source=image_np,
            conf=0.25,
            imgsz=1280,
            max_det=20,
            iou=0.45
        )

    except Exception as e:

        return {

            "success": False,

            "error":

            f"YOLO Error: {str(e)}"
        }


    # =====================================
    # OUTPUT
    # =====================================

    detections = []

    total_kalori = 0

    total_protein = 0

    total_karbo = 0

    total_lemak = 0


    # =====================================
    # PROCESS RESULT
    # =====================================

    detections = []

    detected_names = set()

    for result in results:

        boxes = result.boxes

        print("\n===== DETEKSI YOLO =====")

        for box in boxes:

            cls_id = int(box.cls[0])

            conf = float(box.conf[0])

            class_name = model.names[cls_id]

            print(
                class_name,
                round(conf * 100,2)
            )

        print(
            f"Jumlah objek terdeteksi: {len(boxes)}"
        )

        for box in boxes:

            cls_id = int(
                box.cls[0]
            )

            conf = float(
                box.conf[0]
            )

            class_name = str(
                model.names[cls_id]
            ).strip().lower()

            if conf < 0.25:
                continue

            detected_names.add(
                class_name
            )

            nutrition = get_nutrition(
                class_name
            )

            total_kalori += nutrition["kalori"]
            total_protein += nutrition["protein"]
            total_karbo += nutrition["karbo"]
            total_lemak += nutrition["lemak"]

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detections.append({

                "nama":
                class_name.title(),

                "confidence":
                round(conf * 100, 2),

                "kalori":
                nutrition["kalori"],

                "protein":
                nutrition["protein"],

                "karbo":
                nutrition["karbo"],

                "lemak":
                nutrition["lemak"],

                "bbox": {

                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

            print(
                f"Detected: {class_name} "
                f"({round(conf*100,2)}%)"
            )

    # =====================================
    # STATUS GIZI
    # =====================================

    score = 0

    if total_kalori >= 500:
        score += 25

    if total_protein >= 15:
        score += 25

    if total_karbo >= 65:
        score += 25

    if total_lemak >= 15:
        score += 25


    if score >= 80:

        status = "Bergizi"

    elif score >= 50:

        status = "Kurang Bergizi"

    else:

        status = "Tidak Bergizi"

    # =====================================
    # RESPONSE
    # =====================================

    return {

        "success": True,

        "detections": detections,

        "summary": {

            "total_kalori":
            round(total_kalori, 2),

            "total_protein":
            round(total_protein, 2),

            "total_karbo":
            round(total_karbo, 2),

            "total_lemak":
            round(total_lemak, 2),

            "status":
            status
        }
    }


# =========================================================
# RUN SERVER AUTOMATICALLY FOR DEPLOYMENT
# =========================================================

if __name__ == "__main__":
    import uvicorn
    # Membaca PORT dari environment variable server (Render/Heroku/Koyeb)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)