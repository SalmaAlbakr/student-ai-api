from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# =========================
# Load ML Model
# =========================
model = joblib.load("student_model.pkl")


# =========================
# Recommendations
# =========================
def get_recommendations(weak_skill):
    recommendations_map = {
        "understanding_errors": [
            "راجع المفاهيم الأساسية من الدرس",
            "استخدم خرائط ذهنية للفهم"
        ],
        "recall_errors": [
            "كرر المعلومات أكثر من مرة",
            "استخدم أسئلة مراجعة سريعة"
        ],
        "calculation_errors": [
            "تدرب على خطوات الحل خطوة خطوة",
            "حل مسائل إضافية يوميًا"
        ],
        "attention_errors": [
            "اقرأ السؤال ببطء قبل الحل",
            "تجنب التسرع أثناء الإجابة"
        ],
        "analysis_errors": [
            "تدرب على أسئلة التفكير العليا",
            "حل أسئلة مركبة وتحليلية"
        ]
    }
    return recommendations_map.get(weak_skill, [])


skills_names = {
    "understanding_errors": "الفهم",
    "recall_errors": "التذكر",
    "calculation_errors": "الحساب",
    "attention_errors": "التركيز",
    "analysis_errors": "التحليل"
}


# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return "API is running"


# =========================
# Predict Route
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = pd.DataFrame([{
            "accuracy": data["accuracy"],
            "understanding_errors": data["understanding_errors"],
            "recall_errors": data["recall_errors"],
            "calculation_errors": data["calculation_errors"],
            "attention_errors": data["attention_errors"],
            "analysis_errors": data["analysis_errors"],
        }])

        final_level = model.predict(features)[0]

        skills = {
            "understanding_errors": data["understanding_errors"],
            "recall_errors": data["recall_errors"],
            "calculation_errors": data["calculation_errors"],
            "attention_errors": data["attention_errors"],
            "analysis_errors": data["analysis_errors"],
        }

        weakest_skill = max(skills, key=skills.get)
        weakest_name = skills_names.get(weakest_skill, "غير معروف")

        recommendations = get_recommendations(weakest_skill)

        return jsonify({
            "final_level": str(final_level),
            "weakest_skill": weakest_skill,
            "weak_area_message": f"أنت تحتاج تحسين مهارة {weakest_name}",
            "main_tip": recommendations[0] if recommendations else "استمر في التدريب",
            "recommendations": recommendations,
            "scores": data
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# Run (Railway safe)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

    