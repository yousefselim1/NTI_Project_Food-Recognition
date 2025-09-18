

import os, json, numpy as np, requests, re
import streamlit as st
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


GEMINI_API_KEY = ""   
GENERATOR_MODEL_PRIMARY = "gemini-2.0-flash"
REQUEST_TIMEOUT_SECS = 65

EXPORT_DIR   = Path("export")
MODEL_PATH   = EXPORT_DIR / "qiima_nutrition_regressor.keras"
WEIGHTS_PATH = EXPORT_DIR / "qiima_nutrition_regressor.weights.h5"
STATS_PATH   = EXPORT_DIR / "target_stats.json"


# 2) UI

st.set_page_config(page_title="قِيمَة | Qiima", page_icon="⚠️", layout="centered")
st.title("قِيمَة (Qiima) — تقدير التغذية + تحذيرات ذكية")
st.caption("التقديرات تقريبية وليست بديلاً عن الاستشارة الطبية.")

with st.sidebar:
    st.header("معلومات الطبق (اختياري)")
    dish_name = st.text_input("اسم الطبق:", "")
    user_ingredients = st.text_area("مكوّنات معروفة:", placeholder="مثال: أرز، دجاج، سمسم، حليب...")


def force_rgb_and_resize(img: Image.Image, size=(384, 384)) -> np.ndarray:
    """Ensure 3-ch RGB & fixed size; returns [1,H,W,3] in [0,1]."""
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.expand_dims(arr, 0)

def denorm_and_exp(pred_norm, stats):
    z_mean = stats.get("z_mean", {}); z_std = stats.get("z_std", {})
    cols   = stats.get("cols", ["kcal","carbs","protein","fat"])
    zmean = np.array([z_mean[c] for c in cols], dtype="float32")
    zstd  = np.array([z_std[c]  for c in cols], dtype="float32")
    z = pred_norm * zstd + zmean
    y = np.expm1(z)
    return np.clip(y, 0, None)

ALLERGEN_MAP = {
    "الفول السوداني": ["فول سوداني","فول_سوداني","peanut","peanuts"],
    "المكسرات الشجرية": ["لوز","جوز","عين الجمل","بندق","فستق","كاجو","فستق حلبي","almond","walnut","hazelnut","pistachio","cashew","pecan"],
    "الحليب/الألبان": ["حليب","لبن","زبادي","جبن","زبدة","dairy","milk","yogurt","cheese","butter","whey","casein"],
    "البيض": ["بيض","egg","eggs","albumen"],
    "السمك": ["سمك","fish","salmon","tuna","cod","sardine","mackerel"],
    "الصدفيات": ["روبيان","جمبري","قريدس","سلطعون","كركند","محار","shrimp","prawn","crab","lobster","shellfish","oyster","mussel","clam"],
    "القمح/الغلوتين": ["قمح","جلوتين","gluten","wheat","flour","bread","pasta","semolina"],
    "الصويا": ["صويا","فول الصويا","soy","tofu","soybean","edamame","tempeh"],
    "السمسم": ["سمسم","طحينة","tahini","sesame"],
}
def detect_allergens(text: str) -> list:
    if not text: return []
    t = text.lower(); found = []
    for label, keys in ALLERGEN_MAP.items():
        for k in keys:
            if re.search(r"\b" + re.escape(k.lower()) + r"\b", t):
                found.append(label); break
    return sorted(set(found))

def derive_risk_flags(kcal: float, carbs: float, protein: float, fat: float) -> dict:
    return {
        "high_kcal":   kcal   >= 700,
        "high_carbs":  carbs  >= 60,
        "high_fat":    fat    >= 30,
        "very_high_protein": protein >= 60,
    }

def call_gemini_warnings(dish_name: str, user_ingredients: str, preds: dict, allergens: list, flags: dict):
    if not GEMINI_API_KEY:
        return "⚠️ لا يوجد مفتاح Gemini داخل الملف."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATOR_MODEL_PRIMARY}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        "أنت مساعد سلامة غذائية عربي. أعد نقاط تحذير فقط (بدون مقدمة/خاتمة)، قصيرة وواضحة.\n\n"
        "البيانات:\n"
        f"- الاسم: {dish_name or 'غير معروف'}\n"
        f"- المكوّنات المذكورة: {user_ingredients or 'غير مذكورة'}\n"
        f"- القيم التقريبية: السعرات={preds.get('kcal',0):.0f} ك.سعر، "
        f"كربوهيدرات={preds.get('carbs',0):.1f} غ، بروتين={preds.get('protein',0):.1f} غ، دهون={preds.get('fat',0):.1f} غ.\n"
        f"- مسببات حساسية محتملة: {', '.join(allergens) if allergens else 'غير محدد'}\n"
        f"- أعلام الإفراط: {', '.join([k for k,v in flags.items() if v]) or 'لا يوجد'}\n\n"
        "التعليمات:\n"
        "• اذكر مسببات الحساسية المحتملة بوضوح مع التنبيه للتعرّض المتبادل.\n"
        "• نبّه مرضى السكري عند ارتفاع الكربوهيدرات (تحكم حصص/كربوهيدرات).\n"
        "• نبّه عند ارتفاع الدهون وكثافة الطاقة.\n"
        "• نبّه مرضى الكلى عند ارتفاع البروتين.\n"
        "• أضف تحذيرات عامة للحمل والحساسية غير المعروفة.\n"
        "• اختم بجملة قصيرة: التقديرات تقريبية وليست نصيحة طبية.\n"
    )
    data = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.2,"maxOutputTokens":400}}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=REQUEST_TIMEOUT_SECS)
        r.raise_for_status()
        out = r.json()
        try:
            return out["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"تعذّر الاتصال بـ Gemini: {e}"


#  model
def build_qiima_arch(img_size=(384,384)):
    base = keras.applications.EfficientNetB3(
        include_top=False, input_shape=img_size + (3,), weights=None
    )
    x_in = keras.Input(shape=img_size + (3,), name="image_rgb")
    x = keras.applications.efficientnet.preprocess_input(x_in)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(4, activation="linear", name="nutrients")(x)
    return keras.Model(x_in, out, name="qiima_efficientnet_b3_384")

@st.cache_resource
def load_model_and_stats():
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Missing {STATS_PATH}.")
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if MODEL_PATH.exists():
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
            st.success(f"تم تحميل نموذج .keras: {MODEL_PATH.name}")
            return model, stats
        except Exception as e:
            st.warning(f"فشل تحميل .keras (سيتم استخدام الخطة البديلة): {e}")

    model = build_qiima_arch((384,384))
    if WEIGHTS_PATH.exists():
        model.load_weights(WEIGHTS_PATH)
        st.success(f"تم تحميل الأوزان: {WEIGHTS_PATH.name}")
        return model, stats

    raise FileNotFoundError("لا يوجد .keras صالح ولا weights.h5.")


# App flow

try:
    model, stats = load_model_and_stats()
except Exception as e:
    st.error("تعذّر تحميل النموذج/الإحصاءات. تأكّد من وجود export/* (وأعد التصدير من الدفتر v3 إذا لزم).")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader("ارفع صورة الطبق (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="الصورة المدخلة", use_container_width=True)

    x = force_rgb_and_resize(img, size=(384,384))
    pred_norm = model.predict(x, verbose=0)
    y = denorm_and_exp(pred_norm, stats)[0]   # 
    kcal, carbs, protein, fat = y

    
    st.subheader("التقديرات")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("السعرات (ك.سعر)", f"{kcal:.0f}")
        st.metric("بروتين (غ)", f"{protein:.1f}")
    with col2:
        st.metric("كربوهيدرات (غ)", f"{carbs:.1f}")
        st.metric("دهون (غ)", f"{fat:.1f}")

    # --- فحص الاتساق 4/4/9 ---
    kcal_from_macros = 4*(carbs + protein) + 9*fat
    diff_pct = abs(kcal_from_macros - kcal) / max(1.0, kcal) * 100
    st.write(f"السعرات المحسوبة من الماكروز (4/4/9): **{kcal_from_macros:.0f}** ك.سعر")
    if diff_pct > 15:
        st.warning("⚠️ السعرات غير متّسقة مع الماكروز (>15%). قد يكون تقدير التركيب غير دقيق.")
    use_macros_kcal = st.checkbox("استخدم السعرات المحسوبة من الماكروز بدلًا من تنبؤ السعرات", value=False)
    if use_macros_kcal:
        kcal = kcal_from_macros

    # --- تحذيرات Gemini فقط ---
    allergens = detect_allergens((dish_name or "") + " " + (user_ingredients or ""))
    flags = derive_risk_flags(kcal, carbs, protein, fat)

    with st.spinner("جارٍ توليد تحذيرات Gemini..."):
        warnings_txt = call_gemini_warnings(
            dish_name=dish_name.strip(),
            user_ingredients=user_ingredients.strip(),
            preds={"kcal": float(kcal), "carbs": float(carbs), "protein": float(protein), "fat": float(fat)},
            allergens=allergens,
            flags=flags,
        )
    st.subheader("تحذيرات (Gemini)")
    st.write(warnings_txt)

st.markdown("---")
st.caption("© Qiima — Research prototype.")
