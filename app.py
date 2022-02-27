from flask import Flask, request, render_template
import pandas as pd
import joblib



# Declare a Flask app
app = Flask(__name__)

# Main function here

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("diabetes.pkl")
        
        # Get values through input bars
        Gender = request.form.get("Gender")
        AGE = request.form.get("AGE")
        Urea = request.form.get("Urea")

        Cr = request.form.get("Cr")

        HbA1c = request.form.get("HbA1c")

        Chol = request.form.get("Chol")
        HDL = request.form.get("HDL")

        LDL = request.form.get("LDL")

        VLDL = request.form.get("VLDL")
        BMI = request.form.get("BMI")



        
        
        # Put inputs to dataframe
        X = pd.DataFrame([[Gender, AGE,Urea,Cr,HbA1c,Chol,HDL,LDL,VLDL,BMI]], columns = ["Gender", "AGE","Urea","Cr","HbA1c","Chol","HDL","LDL","VLDL","BMI"])
        X.fillna()
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("index.html", output = prediction)
# Running the app
if __name__ == '__main__':
    app.run(debug = True)