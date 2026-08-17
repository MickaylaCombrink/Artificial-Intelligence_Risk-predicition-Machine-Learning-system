"""
Belgium Campus - Academic Risk Assistant
------------------------------------------
Streamlit front-end that connects two things a non-technical user (e.g. an
academic advisor or lecturer) can use without touching any code:

1. A trained scikit-learn model (models/model.pkl) that predicts whether a
   student is Low / Moderate / High academic risk from their engagement data.
2. Gemini, which explains that prediction in plain language and answers
   follow-up questions about it, in a chat window.

The ML model always makes the actual prediction. Gemini is only used to
explain it - it never changes the risk level.

Run with:
    streamlit run interface.py
"""

import streamlit as st

from gemini_assistant import predict_risk, explain_prediction, ask_assistant

st.set_page_config(
    page_title="Belgium Campus - Academic Risk Assistant",
    page_icon="🎓",
    layout="centered",
)

RISK_STYLE = {
    "Low": {"color": "#1a7f37", "bg": "#e6f4ea", "emoji": "🟢"},
    "Moderate": {"color": "#9a6700", "bg": "#fff6e5", "emoji": "🟠"},
    "High": {"color": "#c92a2a", "bg": "#fde8e8", "emoji": "🔴"},
}

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "student_data" not in st.session_state:
    st.session_state.student_data = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)

st.title("🎓 Belgium Campus Academic Risk Assistant")
st.caption(
    "Enter a student's engagement data below. The machine-learning model "
    "predicts an early-warning risk level, and the assistant explains it "
    "in plain language."
)

# ---------------------------------------------------------------------
# Step 1: Student data input -> ML model prediction
# ---------------------------------------------------------------------
with st.form("student_form"):
    st.subheader("1. Student data")

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        quiz_avg = st.slider("Quiz average (%)", 0, 100, 60)
        assignment_avg = st.slider("Assignment average (%)", 0, 100, 60)
    with col2:
        moodle_activity = st.slider("Moodle activity (%)", 0, 100, 60)
        previous_avg = st.slider("Previous average (%)", 0, 100, 60)

    submitted = st.form_submit_button("Predict risk level", use_container_width=True)

if submitted:
    student_data = {
        "Attendance": attendance,
        "QuizAverage": quiz_avg,
        "AssignmentAverage": assignment_avg,
        "MoodleActivity": moodle_activity,
        "PreviousAverage": previous_avg,
    }

    with st.spinner("Running the model..."):
        prediction = predict_risk(student_data)

    st.session_state.student_data = student_data
    st.session_state.prediction = prediction
    st.session_state.explanation = None  # reset, fetch fresh below
    st.session_state.chat_history = []  # new student -> new conversation

# ---------------------------------------------------------------------
# Step 2: Show prediction + Gemini explanation
# ---------------------------------------------------------------------
if st.session_state.prediction:
    prediction = st.session_state.prediction
    style = RISK_STYLE[prediction]

    st.subheader("2. Result")
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:0.5rem;background-color:{style['bg']};
        border:1px solid {style['color']};">
            <span style="font-size:1.3rem;font-weight:600;color:{style['color']};">
                {style['emoji']} {prediction} Risk
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This is an early-warning indicator from the model, not a final "
        "judgement about the student."
    )

    if st.session_state.explanation is None:
        try:
            with st.spinner("Asking the assistant to explain this..."):
                st.session_state.explanation = explain_prediction(
                    prediction, st.session_state.student_data
                )
        except ValueError as e:
            st.session_state.explanation = ""
            st.warning(str(e))

    if st.session_state.explanation:
        st.markdown("**Explanation & suggestions**")
        st.write(st.session_state.explanation)

    # -------------------------------------------------------------
    # Step 3: Follow-up chat about this student's prediction
    # -------------------------------------------------------------
    st.subheader("3. Ask a follow-up question")
    st.caption(
        "e.g. \"Which factor matters most here?\" or \"What support should "
        "be offered first?\""
    )

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    question = st.chat_input("Ask about this student's prediction...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        try:
            with st.spinner("Thinking..."):
                answer = ask_assistant(
                    question,
                    prediction=st.session_state.prediction,
                    student_data=st.session_state.student_data,
                )
        except ValueError as e:
            answer = f"⚠️ {e}"

        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.write(answer)
else:
    st.info("Fill in the student data above and click **Predict risk level** to begin.")

st.divider()
st.caption(
    "Model: scikit-learn classifier trained on historical student engagement "
    "data. Explanations: Gemini. The model always makes the prediction; "
    "Gemini only explains it and never overrides it."
)
