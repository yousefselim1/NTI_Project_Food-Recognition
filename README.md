# 🍽️ Qiima Nutrition Regressor

This project trains a deep learning model to **estimate calories and
macronutrients (carbs, protein, fat)** directly from food images using
the [Nutrition5k
dataset](https://www.kaggle.com/datasets/kmader/nutrition5k) (Kaggle
mirror).

The model is built with **TensorFlow/Keras** using **EfficientNetB3** as
the image backbone.

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── data/                          # Downloaded Nutrition5k Kaggle mirror (via kagglehub)
    ├── export/
    │   ├── qiima_nutrition_regressor.keras   # Full Keras model
    │   ├── qiima_nutrition_regressor.weights.h5 # Weights only
    │   ├── savedmodel/                # TF SavedModel for serving
    │   └── target_stats.json          # Normalization stats (log1p + z-score)
    ├── train.ipynb                    # Training notebook
    └── README.md

------------------------------------------------------------------------

## ⚙️ Setup

### 1. Install dependencies

``` bash
pip install tensorflow tensorflow-addons kagglehub matplotlib
```

### 2. Download dataset

``` python
import kagglehub
path = kagglehub.dataset_download("kmader/nutrition5k")
print("Dataset downloaded to:", path)
```

### 3. Train (optional)

Use `train.ipynb` to: - Preprocess pickle-based images - Normalize
targets with `log1p + z-score` - Train EfficientNetB3 in **two
stages**: - Frozen backbone (warm-up) - Fine-tune top layers with
**cosine restarts**

------------------------------------------------------------------------

## 📊 Results

Validation performance (denormalized):

-   **MAE kcal** ≈ 86\
-   **MAE carbs** ≈ 11 g\
-   **MAE protein** ≈ 10.6 g\
-   **MAE fat** ≈ 8 g\
-   **mMAPE**: kcal \~41%, macros \~52--53%

------------------------------------------------------------------------

## 🔮 Inference

``` python
import json, numpy as np, tensorflow as tf
from tensorflow import keras
from PIL import Image

# --- Load model ---
model = keras.models.load_model("export/qiima_nutrition_regressor.keras")

# --- Load normalization stats ---
with open("export/target_stats.json") as f:
    stats = json.load(f)

mean = np.array(stats["mean"])
std  = np.array(stats["std"])

# --- Preprocess an image ---
def preprocess_image(path, size=384):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

# --- Predict and denormalize ---
def predict(path):
    x = preprocess_image(path)
    y_pred = model.predict(x)[0]

    # denormalize
    y_denorm = (y_pred * std) + mean
    y_values = np.expm1(y_denorm)

    return dict(
        kcal=float(y_values[0]),
        carbs=float(y_values[1]),
        protein=float(y_values[2]),
        fat=float(y_values[3]),
    )

print(predict("example_food.jpg"))
```

------------------------------------------------------------------------

## 🚀 Next Steps

-   Train longer with **early stopping**\
-   Try **larger EfficientNets** (B4/B5)\
-   Add **RandAugment / AutoAug**\
-   Tune loss weights for kcal vs macros
