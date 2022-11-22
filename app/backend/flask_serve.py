import pickle

from flask import Flask, request
from flask_restful import Api, Resource
import numpy as np

# Creating our app and API
app = Flask(__name__)
api = Api(app)


@app.route("/")
def default_endpoint():
    return "Serving Lending Club dataset-trained machine learning models predicting default or no default on given customer observation"


# Get paths of trained sklearn models
qda_path = "/app/app/data/qda_model.pickle"
lda_path = "/app/app/data/lda_model.pickle"
lg1_path = "/app/app/data/lg1.pickle"
gbc_path = "/app/app/data/gbc.pickle"

# Load models
with open(qda_path, "rb") as f:
    QDA = pickle.load(f)

with open(lda_path, "rb") as g:
    LDA = pickle.load(g)

with open(lg1_path, "rb") as i:
    LOGIT = pickle.load(i)

with open(gbc_path, "rb") as j:
    GBC = pickle.load(j)

models = {"QDA": QDA, "LDA": LDA, "LOGIT": LOGIT, "GBC": GBC}


class PredictDefault(Resource):
    """PredictDefault class that will handle queries"""

    def get(self):
        """Returns prediction value of given model"""
        user_query = request.json["query"]
        model = models.get(request.json["model"])
        prediction = model.predict(user_query)
        pred_proba = model.predict_proba(user_query)
        if prediction == 0:
            pred_text = "Default"
        else:
            pred_text = "No Default"
        # Round the predict proba value and set to new variable
        confidence = list(np.round(pred_proba[0], 3))
        # Create JSON object
        output = {"prediction": pred_text, "confidence": confidence}
        return output


api.add_resource(PredictDefault, "/api/v1/predict")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
