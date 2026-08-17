
How it fits together:
- `prediction.py` cleans the raw CSV and produces `cleaned_student_academic_risk_dataset.csv` and plots.
- `trainModel.py` loads the cleaned CSV, trains models, compares results and saves the best model and preprocessing artifacts to `models/`.
- `interface.py` is the user-facing Streamlit app and uses `gemini_assistant.py` to call the saved model and request explanations from Gemini.
- `gemini_assistant.py` loads model artifacts and encapsulates Gemini calls, keeping the ML prediction and explanation responsibilities separate.

## Model details
- Input features used by the training script:
  - Attendance, QuizAverage, AssignmentAverage, MoodleActivity, PreviousAverage
- Target: "Risk Level" (Low, Moderate, High) — label-encoded and saved in `models/label_encoder.pkl`.
- Model selection: compares Logistic Regression, Decision Tree, Random Forest and saves the best performing model (by accuracy) to `models/model.pkl`.
- Outputs produced after training:
  - `models/model.pkl`, `models/scaler.pkl`, `models/label_encoder.pkl`, `models/model_summary.csv`, `models/confusion_matrix.png`, `models/feature_importance.png`

## Data
- Included example datasets:
  - `student_academic_risk_dataset_100.csv` — small sample dataset used for EDA.
  - `cleaned_student_academic_risk_dataset.csv` — cleaned dataset used for training (produced by `prediction.py`).
- The repository does not contain any private or PII-protected data. If you substitute real student data, ensure you follow applicable privacy and institutional policies.

## Configuration & secrets
- GEMINI_API_KEY is required for natural-language explanations via Google Gemini. Add it to `.env` (do not commit):
  - Copy `.env.example` → `.env`
  - Edit `.env`:
    - GEMINI_API_KEY=AIzaSy...your_real_key...
- The code loads the API key with python-dotenv. If `GEMINI_API_KEY` is not set, the model prediction still runs, but explanations and chat will raise an error.

## Development notes & suggestions
- Recommended Python: 3.10+
- The project uses scikit-learn (1.9.0 pinned) and Streamlit for the UI.
- To reproduce results locally:
  1. Create venv, install requirements.
  2. Run `python prediction.py` → generate cleaned CSV if not already present.
  3. Run `python trainModel.py` → generates `models/`.
  4. Run `streamlit run interface.py`.
- Error handling: gemini_assistant raises a helpful ValueError if the GEMINI_API_KEY is missing.
- Consider adding a small test suite (pytest) around model training and prediction to guard against regressions.

## Contributing
- Please open issues or pull requests for bug fixes or improvements.
- Before submitting changes that modify the model pipeline, confirm reproducibility by:
  - including a fixed random seed where appropriate (the scripts already use random_state=5)
  - documenting dataset changes and expected metrics

## Known improvements / TODO
- Add a LICENSE file (MIT / Apache / choose appropriate license).
- Add CI to run linting and tests, and a simple check that training script runs (smoke test).
- Add unit tests for predict_risk() and for the data cleaning pipeline.
- Add an example of expected `models/` artifacts or release a trained model for demo users.

## License
Add a LICENSE file to this repository to state how the code may be used. (No license file included in the repo yet.)

## Contact
Repository owner: MickaylaCombrink
(Use GitHub issues or pull requests for questions and suggestions.)
