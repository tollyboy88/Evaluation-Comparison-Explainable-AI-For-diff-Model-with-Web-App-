from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Load the best model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Placeholder for training data, should be loaded accordingly
# Load the actual training data used to fit the model
X_train = pd.read_csv("X_train.csv")  # Update with correct path

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, X_train)


def generate_shap_force_plot(input_data):
    """Generates a SHAP force plot and returns it as a base64 string."""
    shap_values = explainer.shap_values(input_data)
    shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    base_value = explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value
    
    shap.force_plot(
        base_value, 
        shap_values_class1[0, :], 
        input_data, 
        show=False, 
        matplotlib=True
    )
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']  # Ensure correct order

    input_data = [float(request.form[feature]) for feature in feature_names]

    input_data = np.array(input_data).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] if hasattr(model, 'predict_proba') else None
    shap_plot = generate_shap_force_plot(input_data)
    
    # File paths for additional visualizations
    images = ['cm_female.png', 'cm_male.png', 'feature_importance.png', 'RandomForest_cm.png', 'shap_summary.png']
    image_data = {}
    
    for img in images:
        if os.path.exists(f'static/{img}'):
            with open(f'static/{img}', 'rb') as f:
                image_data[img] = base64.b64encode(f.read()).decode('utf-8')
    
    # Read SHAP explanation text
    explanation_text = ""
    if os.path.exists('static/shap_explanation.txt'):
        with open('static/shap_explanation.txt', 'r') as f:
            explanation_text = f.read()
    
    return render_template(
        'result.html', 
        prediction=prediction, 
        probability=probability, 
        shap_plot=shap_plot, 
        explanation=explanation_text,
        images=image_data
    )

if __name__ == '__main__':
    app.run(debug=True)
