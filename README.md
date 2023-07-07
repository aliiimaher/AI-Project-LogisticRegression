# AI - Implementing Logistic Regression for Diabetes Labeling

Authors: [Mehdi Mardani](@mahdi712), Ali Maher
Date: 1402/04/11

## Introduction

This project focuses on implementing logistic regression for the purpose of labeling diabetes based on patient attributes. The goal is to train a model that can accurately predict whether a person is diabetic or not based on features such as glucose, blood pressure, skin thickness, and BMI.

## Approach

### Data Cleaning

Before training the model, the data is cleaned by replacing invalid records with the means of each feature. Specifically, zero values in the features Glucose, BloodPressure, SkinThickness, and BMI are replaced with their respective means.

### Model Training

- **Weights and Bias Initialization:**
    - Weights (w1, w2, ..., wn) are initialized as zero.
    - Bias (b or theta_0) is initialized as zero.

- **Training Process:**
    - Given a data point, the model predicts the result using the sigmoid function: y = 1 / (1 + e^(-wx + b)).
    - The error is calculated as the difference between the predicted value (y_hat) and the actual value (y): error = y - y_hat.
    - Gradient descent is used to update the weights and bias:
        - w1 = w1 + x1 * error * learning_rate (and similarly for other weights)
        - b = b + error * learning_rate
    - This process is repeated for a specified number of iterations (n_iters).

### Testing

- Given a data point, the model calculates the predicted value (y_hat) by putting the feature values into the sigmoid function.
- The label is chosen based on the value of y_hat: if y_hat > 0.5, the label is set to 1 (indicating diabetes), otherwise it is set to 0 (indicating non-diabetes).

### Main Program

The main program provides a simple interface with the following options:

- Accuracy Calculation: Calculate the accuracy of the logistic regression model.
- Prediction: Predict if a person has diabetes or not based on their feature values.
- Exit: Exit the program.

## Data Details

- Total Data: 2728 samples
- Train Data: Approximately 70% of the total data (1909 samples)
- Test Data: Remaining 30% of the total data (819 samples)

By following this approach, we aim to train a logistic regression model that can accurately label diabetes based on patient attributes.
