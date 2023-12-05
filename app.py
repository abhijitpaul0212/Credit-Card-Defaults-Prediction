# app.py

from flask import Flask, request, render_template, jsonify
from src.CreditCardDefaultsPrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.CreditCardDefaultsPrediction.components.model_trainer import ModelTrainer

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    """
    Prediction method
    """
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data = CustomData(            
            limit_balance=float(request.form.get('limit_balance')),
            sex=int(request.form.get('sex')),
            education=int(request.form.get('education')),
            marriage=int(request.form.get('marriage')),
            age=int(request.form.get('age')),
            pay_sept=int(request.form.get('pay_sept')),
            pay_aug=int(request.form.get('pay_aug')),
            pay_jul=int(request.form.get('pay_jul')),
            pay_jun=int(request.form.get('pay_jun')),
            pay_may=int(request.form.get('pay_may')),
            pay_apr=int(request.form.get('pay_apr')),
            bill_amount_sept=float(request.form.get('bill_amount_sept')),
            bill_amount_aug=float(request.form.get('bill_amount_aug')),
            bill_amount_jul=float(request.form.get('bill_amount_jul')),
            bill_amount_jun=float(request.form.get('bill_amount_jun')),
            bill_amount_may=float(request.form.get('bill_amount_may')),
            bill_amount_apr=float(request.form.get('bill_amount_apr')),
            pay_amount_sept=float(request.form.get('pay_amount_sept')),
            pay_amount_aug=float(request.form.get('pay_amount_aug')),
            pay_amount_jul=float(request.form.get('pay_amount_jul')),
            pay_amount_jun=float(request.form.get('pay_amount_jun')),
            pay_amount_may=float(request.form.get('pay_amount_may')),
            pay_amount_apr=float(request.form.get('pay_amount_apr'))
        )
        final_data = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)
        result = "DEFAULT" if pred[0] == 1 else "NOT DEFAULT"

        return render_template("result.html", final_result=result)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
