

---

# Final Model Cleaned

This repository contains a comprehensive Python script for data cleaning, exploratory analysis, model training, evaluation, and saving of regression models aimed at predicting flood probability. Additionally, it includes an interactive UI dashboard built with Streamlit, which can be hosted via ngrok for easy access.

## Overview

The repository includes:

- **Data Processing & Model Training:**  
  - **Data Loading & Cleaning:** Loads a CSV dataset, checks for missing values and duplicate rows, and handles outliers via capping.  
  - **Exploratory Data Analysis (EDA):** Generates boxplots and a heatmap to visualize feature distributions and correlations.  
  - **Feature Engineering & Preprocessing:** Selects relevant numeric features, drops unnecessary columns (e.g., `id`), and scales the data using standardization.  
  - **Model Building & Evaluation:** Trains multiple regression models, including Linear Regression, Random Forest Regressor, and Polynomial Regression (Degrees 2 and 3).  
  - **Overfitting Check:** Compares training and test performance to assess potential overfitting.  
  - **Model & Scaler Persistence:** Saves the best performing polynomial regression model and the scaler using `joblib` for future inference.

- **Interactive UI Dashboard:**  
  An interactive UI dashboard is implemented using Streamlit. This dashboard allows users to interact with the model and visualize predictions in real-time. The dashboard can be hosted via ngrok to provide remote access.

## Dependencies

Ensure you have the following Python packages installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- streamlit
- ngrok (for tunneling, if required)

You can install the dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

For ngrok, follow the installation instructions on [ngrok's website](https://ngrok.com/).

## Dataset

The script expects a CSV file (e.g., `train.csv`) containing the dataset with features that include an `id` column (optional) and a target column named `FloodProbability`. Update the file paths as needed in the script.

## How to Run

### Model Training Pipeline

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Place the Dataset:**

   Ensure your dataset (e.g., `train.csv`) is in the appropriate directory or update the file path in the script accordingly.

3. **Run the Script:**

   ```bash
   python final_model_cleaned.py
   ```

   This will execute all steps from data cleaning to model evaluation, and save the trained model (`final_poly_model_degree3.pkl`) and scaler (`scaler.pkl`) in the working directory.

### Running the Interactive UI Dashboard

1. **Navigate to the Dashboard File:**

   The repository includes a separate code file for the Streamlit dashboard.

2. **Run the Dashboard:**

   ```bash
   streamlit run your_dashboard_file.py
   ```

3. **Hosting with ngrok (Optional):**

   To expose your dashboard externally, run ngrok on the port provided by Streamlit (default is 8501):

   ```bash
   ngrok http 8501
   ```

   Follow the instructions provided by ngrok to access your dashboard from a public URL.

## Code Structure & Explanation

- **Data Preprocessing & EDA:**  
  The script performs data cleaning, handles outliers, visualizes feature distributions, and creates a correlation heatmap.

- **Feature Scaling & Model Training:**  
  The data is split into training and test sets, features are scaled, and various regression models are trained and evaluated using MSE and R² Score.

- **Overfitting Check & Model Persistence:**  
  The code includes an overfitting check and saves the best performing model and scaler.

- **Interactive Dashboard:**  
  The dashboard file (provided separately) uses Streamlit to create an interactive interface, allowing users to input parameters and view model predictions dynamically.

## Future Improvements

- Hyperparameter tuning for models such as Random Forest and SVR.
- Cross-validation to ensure model robustness.
- Additional feature engineering based on domain knowledge.
- Enhancements to the UI dashboard for improved user experience.

---

