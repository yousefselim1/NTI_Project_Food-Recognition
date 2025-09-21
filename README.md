# 🍽️ Qiima Food Recognition & Nutrition Estimator

The **Qiima Nutrition Regressor** is a deep learning project that
predicts **calories and macronutrients (carbohydrates, protein, fat)**
directly from food images.\
It uses the [Nutrition5k
dataset](https://www.kaggle.com/datasets/kmader/nutrition5k) (Kaggle
mirror) and is built with **TensorFlow/Keras** using **EfficientNetB3**
as the backbone.

------------------------------------------------------------------------

## ✨ Features

-   Predicts **Calories, Carbs, Protein, and Fat** from a single food
    image.
-   Uses **EfficientNetB3** for image feature extraction.
-   Two-stage training strategy:
    1.  Warm-up with frozen backbone\
    2.  Fine-tuning with cosine learning rate restarts\
-   Supports **TF SavedModel** and **Keras model export** for
    deployment.\
-   Includes preprocessing and normalization pipeline.

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── data/                          # Downloaded Nutrition5k Kaggle mirror (via kagglehub)
    ├── export/
    │   ├── qiima_nutrition_regressor.keras      # Full Keras model
    │   ├── qiima_nutrition_regressor.weights.h5 # Weights only
    │   ├── savedmodel/                 # TF SavedModel for serving
    │   └── target_stats.json           # Normalization stats (log1p + z-score)
    ├── train.ipynb                     # Training notebook
    └── README.md

------------------------------------------------------------------------

## ⚙️ Setup

### 1. Clone the repo & install dependencies

``` bash
git clone <your-repo-url>
cd qiima-nutrition
pip install -r requirements.txt
```

### 2. Download the dataset

``` python
import kagglehub
path = kagglehub.dataset_download("kmader/nutrition5k")
print("Dataset downloaded to:", path)
```

### 3. Training (optional)

Open `train.ipynb` to: - Load dataset (pickle-based images) - Normalize
targets with `log1p + z-score` - Train EfficientNetB3 in two stages -
Export models and normalization stats

------------------------------------------------------------------------

## 📊 Model Performance

Validation results (denormalized):

  Metric        Value
  ------------- ------------------------------
  MAE kcal      \~86 kcal
  MAE carbs     \~11 g
  MAE protein   \~10.6 g
  MAE fat       \~8 g
  mMAPE         kcal \~41%, macros \~52--53%

------------------------------------------------------------------------

## 🔮 Inference Example

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
-   Try larger EfficientNets (B4/B5)\
-   Add stronger data augmentations (RandAugment / AutoAug)\
-   Tune **loss weights** between kcal and macros

------------------------------------------------------------------------

## 📜 License

This project is for **research and educational purposes only**.\
Do not use predictions for medical or dietary decisions without
professional validation.

![5942615509679853014_121](https://github.com/user-attachments/assets/59db3dc2-bf7c-4ec2-8e9f-6d14ee9826f8)


