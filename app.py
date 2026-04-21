from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    # Debug (optional - remove later)
    print(form_data)

    try:
        # Safe fetching (handles both cases)
        pclass = int(form_data.get('Pclass') or form_data.get('pclass'))
        age = float(form_data.get('Age') or form_data.get('age'))
        fare = float(form_data.get('Fare') or form_data.get('fare'))

        sex_value = form_data.get('Sex') or form_data.get('sex')
        sex = 0 if sex_value == 'male' else 1

    except Exception as e:
        return f"Input Error: {e}"

    # ONLY 4 FEATURES (your model expects this)
    final_input = np.array([[pclass, sex, age, fare]])

    prediction = model.predict(final_input)

    result = "Survived" if prediction[0] == 1 else "Not Survived"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
