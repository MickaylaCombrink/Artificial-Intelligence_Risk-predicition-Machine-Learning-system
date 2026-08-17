import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Directory this file lives in, so the app works no matter which folder
# Streamlit is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

# The Gemini client is created lazily (see get_client) so that importing
# this module - and using the ML model on its own - never fails just
# because a GEMINI_API_KEY hasn't been configured yet. The key is only
# required when explain_prediction() or ask_assistant() is actually called.
_client = None


def get_client():
    """Return a cached Gemini client, creating it on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. Add it to a .env file "
                "(GEMINI_API_KEY=your_key_here) or set it as an "
                "environment variable."
            )
        _client = genai.Client(api_key=api_key)
    return _client

SYSTEM_PROMPT = """
You are an Academic Risk Support Assistant.

Your purpose is to explain predictions made by an existing machine-learning
model that classifies students into Low Risk, Moderate Risk, or High Risk.

The machine-learning model makes the actual prediction. You must not change
or override the prediction.

Explain predictions using only the information provided. Do not invent
student information or claim that a specific factor definitely caused
the prediction.

If required information is missing, say what information is missing.

The prediction is an academic early-warning indicator and not a final
judgement about a student.

For questions unrelated to academic risk, explain that you are designed
to assist with academic-risk predictions and explanations.

Keep responses clear, professional, and concise.
"""
def predict_risk(student_data):
    features = pd.DataFrame([[
        student_data["Attendance"],
        student_data["QuizAverage"],
        student_data["AssignmentAverage"],
        student_data["MoodleActivity"],
        student_data["PreviousAverage"]
    ]], columns=[
        "Attendance",
        "QuizAverage",
        "AssignmentAverage",
        "MoodleActivity",
        "PreviousAverage"
    ])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    risk_mapping = {
        0: "Low",
        1: "Moderate",
        2: "High"
    }

    return risk_mapping[prediction]

def explain_prediction(prediction, student_data):
    prompt = f"""
{SYSTEM_PROMPT}

The machine-learning model produced this prediction:

Prediction: {prediction}

Student data:
{student_data}

Explain this prediction using the available information.
Provide one or two practical academic-support suggestions.
"""

    response = get_client().models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text


def ask_assistant(question, prediction=None, student_data=None):
    context = ""

    if prediction is not None:
        context += f"Current ML prediction: {prediction}\n"

    if student_data is not None:
        context += f"Student data: {student_data}\n"

    prompt = f"""
{SYSTEM_PROMPT}

{context}

User question:
{question}
"""

    response = get_client().models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text

if __name__ == "__main__":
    student_data = {
        "Attendance": 80,
        "QuizAverage": 49,
        "AssignmentAverage": 38,
        "MoodleActivity": 55,
        "PreviousAverage": 55
    }

    prediction = predict_risk(student_data)

    print("Prediction:", prediction)

    question = input("\nAsk the assistant: ")

    response = ask_assistant(
        question,
        prediction,
        student_data
    )

    print("\nGemini:")
    print(response)