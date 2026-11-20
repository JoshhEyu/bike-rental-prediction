# Bike Rental Demand Prediction

## 1. Project Overview

This project uses Machine Learning regression techniques to predict hourly bike rental demand based on historical rental data and environmental conditions.

The project explores multiple regression algorithms, compares their performance, and develops a final Gradient Boosting model.

A Streamlit web application provides an interactive interface for making bike rental demand predictions.

## 2. Project Objectives

- Explore and understand bike rental demand data.
- Perform data cleaning and preprocessing.
- Analyze relationships between environmental conditions and rental demand.
- Train and compare multiple regression models.
- Optimize model performance through hyperparameter tuning.
- Evaluate the final model using regression metrics.
- Deploy an interactive prediction application using Streamlit.

## 3. Dataset

The project uses the UCI Bike Sharing Dataset, specifically the hourly rental dataset (`hour.csv`).

The dataset contains information such as:

- Date and time
- Season
- Weather conditions
- Temperature
- Humidity
- Wind speed
- Casual rentals
- Registered rentals
- Total rental demand (`cnt`)

The target variable is `cnt`, representing the total number of bike rentals.

## 4. Machine Learning Models

The project explores several regression algorithms:

- Univariate Linear Regression
- Bivariate Linear Regression
- Multivariate Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

The final selected model is Gradient Boosting.

## 5. Final Model Performance

| Metric | Test Result |
|---|---:|
| MAE | 71.60 |
| RMSE | 99.42 |
| R² Score | 0.7864 |

The final model achieved an R² score of approximately 78.64% on the test dataset.

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 7. Project Structure

```text
Bike_Rental_Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── final_bike_rental_model.pkl
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── evaluation reports
│
├── src/
│   ├── analyze_residuals.py
│   ├── evaluate_final_model.py
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── train_regression_models.py
│   ├── train_advanced_regression_models.py
│   ├── tune_regression_models.py
│   └── visualization scripts
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 8. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/bike-rental-prediction.git
```

Navigate into the project directory:

```bash
cd bike-rental-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 9. Run the Streamlit Application

Execute:

```bash
python -m streamlit run app/app.py
```

Open the local address displayed in your terminal, usually:

```text
http://localhost:8501
```

## 10. Model Deployment

The application loads the saved final model from:

```text
models/final_bike_rental_model.pkl
```

The Streamlit application allows users to provide input features and receive predicted bike rental demand.

## 11. Limitations

- Predictions depend on the quality and characteristics of the training dataset.
- Unusual weather conditions or unexpected events may affect actual rental demand.
- Model performance on unseen data may differ from the reported test results.

## 12. Disclaimer

This project was developed for educational and machine learning demonstration purposes. Predictions are estimates and should not be treated as guaranteed future rental demand.

---

**Developed as a Machine Learning Regression Project.**

## 13. Live Demo

Try the deployed application:

[Open Bike Rental Demand Prediction App](https://bike-rental-prediction-qbtpaapqmjjsk6qblbkvqr.streamlit.app/)

## 14. Author & Copyright

**Project Author:** Eyasu Kebede

© 2026 Eyasu Kebede. All Rights Reserved.

This project is made available for viewing and personal or educational study only. Copying, redistribution, modification for distribution, and commercial use of the original project materials require prior written permission from the copyright holder.

For further details, see the [LICENSE](LICENSE) file.