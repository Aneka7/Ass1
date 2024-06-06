from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import io
import base64

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('cap1.csv')
df = df.replace({'Yes': 1, 'No': 0, 'Maybe': 0.5})
df_encoded = pd.get_dummies(df, columns=['Age', 'Gender', 'Occupation', 'Days_Indoors', 'Mood_Swings'])

# Define features and target variables
X = df_encoded.drop(columns=['Mental_Health_History'])
y = df_encoded['Mental_Health_History']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)
linear_cv_scores = cross_val_score(linear_model, X_scaled, y, cv=10, scoring='r2')
linear_mean_cv_score = np.mean(linear_cv_scores)

# Logistic Regression Model
# For logistic regression, we need to convert the target variable to binary classification
y_binary = df_encoded['Mental_Health_History'].apply(lambda x: 1 if x > 0 else 0)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train_log, y_train_log)
y_pred_logistic = logistic_model.predict(X_test_log)
logistic_accuracy = accuracy_score(y_test_log, y_pred_logistic)
logistic_cv_scores = cross_val_score(logistic_model, X_scaled, y_binary, cv=10, scoring='accuracy')
logistic_mean_cv_score = np.mean(logistic_cv_scores)

# Random Forest Model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_log, y_train_log)
y_pred_rf = random_forest_model.predict(X_test_log)
random_forest_accuracy = accuracy_score(y_test_log, y_pred_rf)
random_forest_cv_scores = cross_val_score(random_forest_model, X_scaled, y_binary, cv=10, scoring='accuracy')
random_forest_mean_cv_score = np.mean(random_forest_cv_scores)

@app.route('/')
def index():
    return render_template('index.html', 
                           linear_mse=linear_mse, linear_r2=linear_r2, linear_mean_cv_score=linear_mean_cv_score,
                           logistic_accuracy=logistic_accuracy, logistic_mean_cv_score=logistic_mean_cv_score,
                           random_forest_accuracy=random_forest_accuracy, random_forest_mean_cv_score=random_forest_mean_cv_score)

@app.route('/plot')
def plot():
    # Plot actual vs. predicted values for linear regression
    plt.figure(figsize=(18, 6))
    
    # Linear Regression Plot
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_linear, alpha=0.5, label='Data points')
    plt.xlabel("Actual Mental Well-being")
    plt.ylabel("Predicted Mental Well-being")
    plt.title("Linear Regression: Actual vs. Predicted Mental Well-being")
    line_x = np.linspace(min(y_test), max(y_test), 100)
    line_y = line_x
    plt.plot(line_x, line_y, color='red', linewidth=2, label='Regression line')
    plt.legend()
    
    # Logistic Regression Plot
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_test_log, logistic_model.predict_proba(X_test_log)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression: ROC Curve')
    plt.legend()

    # Random Forest Plot
    plt.subplot(1, 3, 3)
    fpr_rf, tpr_rf, _ = roc_curve(y_test_log, random_forest_model.predict_proba(X_test_log)[:, 1])
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, color='green', label=f'ROC curve (area = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest: ROC Curve')
    plt.legend()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('plot.html', plot_url=plot_url)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = request.form.to_dict()
        
        # Convert form data to a DataFrame
        input_data = pd.DataFrame([data])
        
        # Convert categorical variables to numerical for prediction
        input_data = input_data.replace({'Yes': 1, 'No': 0, 'Maybe': 0.5})
        input_data = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict using linear regression
        linear_prediction = linear_model.predict(input_data_scaled)[0]
        
        # Predict using logistic regression
        logistic_prediction_proba = logistic_model.predict_proba(input_data_scaled)[0][1]
        logistic_prediction = logistic_model.predict(input_data_scaled)[0]

        # Predict using random forest
        random_forest_prediction_proba = random_forest_model.predict_proba(input_data_scaled)[0][1]
        random_forest_prediction = random_forest_model.predict(input_data_scaled)[0]
        
        return render_template('result.html', 
                               linear_prediction=linear_prediction, 
                               logistic_prediction=logistic_prediction, 
                               logistic_prediction_proba=logistic_prediction_proba,
                               random_forest_prediction=random_forest_prediction,
                               random_forest_prediction_proba=random_forest_prediction_proba)
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
