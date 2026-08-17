# Belgium Campus - Academic Risk Assistant

## Setup
1. python -m venv .venv
2. Activate it: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Mac/Linux)
3. pip install -r requirements.txt
4. Copy .env.example to .env and add your real GEMINI_API_KEY (get one free, no card, at aistudio.google.com)
5. streamlit run interface.py

## Files
- interface.py        - Streamlit web interface (the "question in -> grounded answer out" UI)
- gemini_assistant.py  - Loads the trained model, runs predictions, and asks Gemini to explain them
- trainModel.py        - Trains and evaluates the ML models, saves the best one to models/
- prediction.py        - Data cleaning / EDA script that produced cleaned_student_academic_risk_dataset.csv and the plots
- models/              - Trained model, scaler, label encoder, confusion matrix, model summary
- plots/               - Boxplot and histogram EDA plots
- *.csv                - Raw and cleaned datasets

## Do NOT submit
- .venv/, .venv-erich/, .idea/  (recreated automatically, not needed)
- .env  (contains your real API key - keep it local only, submit .env.example instead)
