from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        age = int(data['age'])
        gender = data['gender']
        estimated_salary = float(data['estimated_salary'])
        
        # Assuming the model was trained with gender as 0 for male and 1 for female
        gender_numeric = 0 if gender.lower() == 'male' else 1

        # Create the feature array
        features = np.array([[age, gender_numeric, estimated_salary]])
        
        # Make prediction
        prediction = model.predict(features)
        output = 'Purchased' if prediction[0] == 1 else 'Not Purchased'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)



