from flask import Flask
from flask_restful import reqparse, Api, Resource
import numpy as np  
from flask import request
import pickle 


# Creating our app and API
app = Flask(__name__)
api = Api(app)

@app.route("/news")
def news():
    return render_template('news.html')


#load our trained QDA model 

qda_path = 'qda_model.pickle'
with open(qda_path, 'rb') as f:
    QDA = pickle.load(f)

lda_path = 'lda_model.pickle'
with open(lda_path, 'rb') as g:
    LDA = pickle.load(g)

lg1_path = 'lg1.pickle'
with open(lg1_path, 'rb') as i:
    LOGIT = pickle.load(i)

gbc_path = 'gbc.pickle'
with open(gbc_path, 'rb') as j:
    GBC = pickle.load(j)

# rf1_path = 'finalized_model.sav'
# with open(rf1_path, 'rb') as k:
#     RF = pickle.load(k)

# RF, meta = pyreadstat.read_sav('/Users/philippeheitzmann/NYCDataScienceAcademy/Capstone/DashApp2/finalized_model.sav')
# RF = joblib.load('finalized_model.sav')



    # #loading groupby addr_state, grade data
# with open('test1.sav','rb') as testsav:
#      eda3 = pickle.load(testsav)



#load our x_train_lg and y_train_lg data

# file_path1 = 'x_train_lg.pickle'
# with open(file_path1, 'rb') as f1:
#     x_train_lg = pickle.load(f1)
    
# file_path2 = 'y_train_lg.pickle'
# with open(file_path2, 'rb') as f2:
#     y_train_lg = pickle.load(f2)




#adding 'query' keyword to our parser 
#query will be the data input to the model
parser = reqparse.RequestParser()
parser.add_argument('query',action='append')
parser.add_argument('model')

#creating our PredictDefault class that will handle queries
#implementing our get method that will return the prediction value 

#@app_route('/')
#def

models = {'QDA':QDA, 'LDA':LDA, 'LOGIT':LOGIT, 'GBC':GBC} #, 'SNN':model} #, 'RF':RF

class PredictDefault(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        #json attribute
        user_query = request.json['query']
        model = models.get(request.json['model'])
        # print('user_query',user_query)
        # print('model', model)
        # vectorize the user's query and make a prediction
        #uq_vectorized = model.vectorizer_transform(
        #    np.array([user_query]))
        prediction = model.predict(user_query)
        pred_proba = model.predict_proba(user_query)
        # Output 'Negative' or 'Positive' along with the score
        # if model == 'SNN':
        # 	prediction = model.predict(user_query)[0][0]
        # 	if prediction > 0.5:
        # 		pred_text = 'No Default'
        # 	else:
        # 		pred_text = 'Default'
        # #pick up here
        # else:
        if prediction == 0:
            pred_text = 'Default'
        else:
            pred_text = 'No Default'
            
        # round the predict proba value and set to new variable

        confidence = list(np.round(pred_proba[0], 3))

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}
        
        return output


#creating another endpoint to our API

api.add_resource(PredictDefault, '/')
  
# example of another endpoint
#api.add_resource(PredictRatings, '/ratings')


if __name__ == '__main__':
    app.run(debug = True)

#app.run params
#update host to '0.0.0.0' when running wit Docker 
#host="0.0.0.0", 
#threaded=True, 

# action='append'
# FROM python:3.6

# WORKDIR /app

# COPY requirements.txt /app
# RUN pip install -r requirements.txt

# COPY . /app

# CMD python app.py
# dockerfile build -t rest-api .
# docker build -t rest-api .
# docker run -it -p 5000:5000 rest-api
# app.run(host="0.0.0.0")    

