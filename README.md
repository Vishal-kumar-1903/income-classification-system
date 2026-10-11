# Income Category Prediction System

An end-to-end Machine Learning web application deployed using Streamlit Cloud that classifies candidate income brackets (> Rs 5 Lakhs/Year vs <= Rs 5 Lakhs/Year) using demographic and employment features from the UCI Adult Census dataset.

---

## Live Application Link
Live Demo: https://income-predictor-vk.streamlit.app

---

## Project Team Members

| S.No | ERP ID | Name | Role & Core Contribution |
| :---: | :---: | :--- | :--- |
| 1 | **6605803** | **Vishal Kumar Shrivastava** | **Team Lead & Full-Stack ML Engineer**<br>- End-to-end architecture & project execution<br>- Model development, decision tree pipeline & serialization<br>- Streamlit UI design, feature mapping & cloud deployment |
| 2 | **6606824** | **Anurag Yadav** | **ML Model Trainer**<br>- Model training support & hyperparameter tuning<br>- Baseline evaluation & algorithm comparison |
| 3 | **6605605** | **Abhinav Saraf** | **Documentation & Presentation Lead**<br>- Project report, viva presentation (PPT) & asset preparation<br>- System workflow diagrams & user documentation |

---

## Key Highlights & Architecture
- Algorithm: Decision Tree Classifier with custom unpickling pipeline (~85.6% Test Accuracy).
- Interactive UI: Real-time parameter tuning with custom dark theme, probability progress meters, and dynamic feature importance charts.
- Data Engineering: Simplified categorical mappings (Education, Employment Sector, Job Role) aligned with real-world industry benchmarks.

---

## Tech Stack
- Language: Python
- Libraries: Streamlit, Scikit-Learn, Pandas, NumPy, Joblib
- Deployment: Streamlit Community Cloud & GitHub

---

## Local Setup & Execution

1. Clone the repository:
   git clone https://github.com/Vishal-kumar-1903/income-classification-system.git
   cd income-classification-system

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py