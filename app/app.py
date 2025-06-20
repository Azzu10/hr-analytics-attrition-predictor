from flask import Flask, render_template, request
import pickle
import pandas as pd
import json

app = Flask(__name__)

# Load model, scaler, and feature columns
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Step 1: Get input
        age = int(request.form["age"])
        income = float(request.form["income"])
        overtime = 1 if request.form["overtime"] == "Yes" else 0

        # Step 2: Build input row
        input_dict = {
            "Age": age,
            "MonthlyIncome": income,
            "OverTime": overtime
        }

        input_data = pd.DataFrame(
            [[input_dict.get(col, 0) for col in feature_columns]],
            columns=feature_columns
        )

        # Step 3: Scale input
        input_scaled = scaler.transform(input_data)

        # Step 4: Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][prediction] * 100
        result_label = "🟢 Likely to Stay" if prediction == 0 else "🟠 Likely to Leave"
        result_msg = f"{result_label} (Confidence: {proba:.2f}%)"

        # Step 5: Render result
        return render_template("result.html",
                               result=result_msg,
                               age=age,
                               income=income,
                               overtime="Yes" if overtime else "No")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return f"<h3>❌ Error during prediction:</h3><pre>{str(e)}</pre>"


if __name__ == "__main__":
    app.run(debug=True)
