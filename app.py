from flask import Flask, request, render_template
import pandas as pd
import joblib
from flask_cors import CORS, cross_origin




# Declare a Flask app
app = Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Main function here

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def main():
    
    # If a form is submitted
    if request.method == "POST":

        # Unpickle classifier
        clf = joblib.load("./ds.pkl")
        
        # Get values through input bars
        Gender=request.json["Gender"]
        AGE = request.json["AGE"]
        Urea = request.json["Urea"]
        Cr = request.json["Cr"]
        Chol = request.json["Chol"]
        HDL = request.json["HDL"]
        TG=request.json["TG"]
        LDL = request.json["LDL"]
        VLDL = request.json["VLDL"]
        BMI = request.json["BMI"]
        HbA1c =request.json["HbA1c"]

        
        # Put inputs to dataframe
        X = pd.DataFrame([[Gender, AGE,Urea,Cr,TG,HbA1c,Chol,HDL,LDL,VLDL,BMI]], columns = [ "Gender","AGE","Urea","Cr","TG","HbA1c","Chol","HDL","LDL","VLDL","BMI"])

        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""

    print('dd',prediction)   
    if prediction==1:
        return 'has diabetes'
    elif prediction==0:
        return'dont have diabetes'

    else :
        return 'The patient maybe has diabetes check your doctor'
# Running the app
if __name__ == '__main__':
    app.run(debug = True)