from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#importing ridge regressor and pickle file!
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
scaler_model=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')
#render_template helps in finding that template which we've rendered/picked up from somewhere else, any other file out there! Looks for a template folder and in there look for the file
@app.route("/predictData",methods=['GET','POST'])
def predict_datapoint():
    #neccessary condition to find out whether get or post request
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data)
        return render_template('home.html',results=result[0])
    else:
        #it'll be get
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
    #keep debug=True else have to restart everytime after editing on flask!