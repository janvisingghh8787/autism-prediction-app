Autism Spectrum Disorder Prediction App
 Overview

This project is an end-to-end Machine Learning application that predicts the likelihood of Autism Spectrum Disorder (ASD) based on behavioral and demographic features. The model is deployed as an interactive web application using Streamlit.
---
##Features

* Predicts ASD likelihood using user input
* Interactive UI built with Streamlit
* Uses behavioral screening questionnaire (A1–A10)
* Handles categorical data with encoding
* Displays prediction probability
* Includes disclaimer for medical use
---
## Machine Learning Models Used

* Decision Tree
* Random Forest (Best Model)
* XGBoost
* Support Vector Machine (for comparison)
---
## Model Performance

* **Best Model:** Random Forest
* **Accuracy:** ~92%
* **Techniques Used:**

  * Cross-validation
  * Hyperparameter tuning (RandomizedSearchCV)
  * SMOTE for handling class imbalance
---
## Tech Stack

* Python
* Scikit-learn
* Pandas & NumPy
* Streamlit (for deployment)

---

## Project Structure

```
autism-prediction-app/
│
├── app.py                          # Streamlit application
├── autism_prediction_model.pkl     # Trained ML model
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## Installation & Running Locally

### 1. Clone the repository

```
git clone https://github.com/your-username/autism-prediction-app.git
cd autism-prediction-app
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

##  Deployment

This project is deployed using **Streamlit Cloud**.

👉 (Add your deployed link here after deployment)

---

## Application Workflow

1. User enters personal details (age, gender, etc.)
2. Answers behavioral questions
3. Model processes input
4. Displays prediction:

   * High likelihood of ASD
   * Low likelihood of ASD

---

##  Disclaimer

This application is for educational and research purposes only.
It is **not a medical diagnosis tool**. Please consult a healthcare professional for proper evaluation.

---

##  Authors 

* Janvi Singh and Ojaswini Sood

---

## Future Improvements

* Add SHAP for model explainability
* Improve UI/UX design
* Add more robust dataset
* Deploy using Docker / cloud services

---

## Acknowledgements

* Scikit-learn documentation
* Streamlit community
* Public datasets for ASD screening

---
