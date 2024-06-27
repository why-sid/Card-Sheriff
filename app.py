from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load dataset
def load_data():
    data = pd.read_csv("E:/B.Tech Major Project/Dataset/creditcard.csv")
    data = data.drop_duplicates()
    data.fillna(data.mean(), inplace=True)
    return data

# Load data once
dataset = load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    model_choice = data.get('model')
    balance_data = data.get('balance') == 'true'
    test_size = float(data.get('test_size'))

    # Use the preloaded dataset
    X = dataset.drop("Class", axis=1)
    y = dataset["Class"]

    # Perform label encoding for categorical variables
    label_encoder = LabelEncoder()
    categorical_cols = X.columns
    for col in categorical_cols:
        X[col] = label_encoder.fit_transform(X[col])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if balance_data:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Scale data
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select model
    if model_choice == "XGBClassifier":
        model = XGBClassifier()
    elif model_choice == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == "CatBoostClassifier":
        model = CatBoostClassifier(verbose=0)
    elif model_choice == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=42)
    elif model_choice == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=42)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create confusion matrix plot
    LABELS = ['Normal', 'Fraud']
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': plot_url
    })

if __name__ == '__main__':
    app.run(debug=True)
