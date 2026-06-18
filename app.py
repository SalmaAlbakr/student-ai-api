from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# =========================
# Load ML Model
# =========================
model = joblib.load("student_model.pkl")


# =========================
# Recommendation Engine
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


# =========================
# Skill Names (for explanation)
# =========================
skills_names = {
    "understanding_errors": "الفهم",
    "recall_errors": "التذكر",
    "calculation_errors": "الحساب",
    "attention_errors": "التركيز",
    "analysis_errors": "التحليل"
}


# =========================
# Predict Route
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # =========================
    # Prepare features for ML
    # =========================
    features = pd.DataFrame([{
        "accuracy": data["accuracy"],
        "understanding_errors": data["understanding_errors"],
        "recall_errors": data["recall_errors"],
        "calculation_errors": data["calculation_errors"],
        "attention_errors": data["attention_errors"],
        "analysis_errors": data["analysis_errors"],
    }])

    # =========================
    # ML Prediction
    # =========================
    final_level = model.predict(features)[0]

    # =========================
    # Find weakest skill
    # =========================
    skills = {
        "understanding_errors": data["understanding_errors"],
        "recall_errors": data["recall_errors"],
        "calculation_errors": data["calculation_errors"],
        "attention_errors": data["attention_errors"],
        "analysis_errors": data["analysis_errors"],
    }

    weakest_skill = max(skills, key=skills.get)

    weakest_name = skills_names.get(weakest_skill, "غير معروف")

    # =========================
    # Explanation for student
    # =========================
    weak_area_message = f"أنت تحتاج تحسين مهارة {weakest_name}"

    # =========================
    # Recommendations
    # =========================
    recommendations = get_recommendations(weakest_skill)

    main_tip = recommendations[0] if recommendations else "استمر في التدريب وحاول تحل أسئلة أكثر"

    # =========================
    # Response
    # =========================
    return jsonify({
        "final_level": final_level,
        "weakest_skill": weakest_skill,
        "weak_area_message": weak_area_message,
        "main_tip": main_tip,
        "recommendations": recommendations,
        "scores": data
    })


app = Flask(__name__)

@app.route('/')
def home():
    return "Hello"



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)