from flask import Flask, request, jsonify
from prediction import predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict',methods=['POST'])
def pred_output():
    data = request.get_json()
    Gender = data.get('gender','')
    Age = data.get('age',0)
    Department = data.get('department','')
    Region = data.get('region','')
    Recruitment_channel = data.get('recruitment_channel','')
    Num_training_completed = data.get('no_trainings',0) 
    Previous_year_rating = data.get('prev_year_rating',0)
    KPIs_met = data.get('kpis_met','No')
    Awards_won = data.get('awards_won','No')
    Length_of_service = data.get('service_length',0)
    Education = data.get('education','')
    Avg_training_score = data.get('avg_training_score',0)
    
    result = predict(Gender,Age,Department,Region,Education,Recruitment_channel,Num_training_completed,Previous_year_rating,Length_of_service,KPIs_met,Awards_won,Avg_training_score)
    response_data = {'message':f'The employee is {result}','status':200}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

'''
api url:
http://localhost:5000/predict

data:
{"gender":"m",
"age":24,
"department":"Technology",
"region":"region_26",
"education":"Bachelor's",
"recruitment_channel":"sourcing",
"no_trainings":1,
"prev_year_rating":4,
"service_length":1,
"kpis_met":"Yes",
"awards_won":"No",
"avg_training_score":77}
'''
