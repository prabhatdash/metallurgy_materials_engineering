# Machine Learning Model Comparison and Analysis

This repository contains code to train and evaluate multiple machine learning regression models using an input dataset with various features. The code trains different models, compares their performance, saves the best-performing model, and generates visualizations for better insights.

## Input Dataset

The input dataset should be a CSV file with the following columns:

- `C`: Carbon content.
- `P`: Phosphorus content.
- `S`: Sulfur content.
- `Al`: Aluminum content.
- `Nb`: Niobium content.
- `Ti`: Titanium content.
- `ceq`: Carbon equivalent value.
- `a_temp`: Austenitizing temperature.
- `r`: **Target variable**, representing the response or dependent variable.

Ensure that the dataset is formatted correctly before running the code.

## Models Used

The code trains and evaluates the following machine learning models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- k-Nearest Neighbors Regressor (kNN)

## Features

- **Model Training and Evaluation**: The code splits the dataset into training and testing sets, trains each model, and evaluates their performance based on R² score, Mean Squared Error (MSE), and accuracy.
- **Visualization**: 
  - Actual vs. predicted values are plotted for each model.
  - Performance comparison using bar charts for R², MSE, and accuracy.
  - Correlation heatmap for the input features.
  - Learning curve for the best-performing model.
- **Model Persistence**: The best-performing model is saved to a `.pkl` file for future predictions.

## Usage

1. Clone this repository.
2. Ensure that the required libraries are installed:
    ```bash
    pip install -r requirements.txt
    ```
3. Place the input CSV file (with columns `C`, `P`, `S`, `Al`, `Nb`, `Ti`, `ceq`, `a_temp`, and `r`) in the root directory.
4. Run the script:
    ```bash
    python main.py
    ```

## Results

After running the script:
- Performance metrics for all models are displayed in the console.
- Plots comparing model performance and actual vs. predicted values are generated and saved in the `graphs/` directory.
- The best model is saved as `best_model.pkl` for later use in making predictions.

## Example Prediction

You can use the saved model to make predictions on new data:

```python
import numpy as np
import joblib

# Load the model
model = joblib.load('best_model.pkl')

# Example input for prediction
input_data = np.array([[76.5071, 22.26, 0.01, 0.21, 0.59, 35.0]])  # Replace with your data
predicted_r = model.predict(input_data)
print(f"Predicted r value: {predicted_r[0]}")
