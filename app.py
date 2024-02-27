from flask import Flask, render_template, request,redirect,flash,url_for,session
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('CardioVascular.csv')

# Drop rows with missing values
df = df.dropna()

# Split the data into training and test sets
train_data = df.sample(frac=0.8, random_state=0)
test_data = df.drop(train_data.index)

# Split the training and test sets into X (input features) and y (output variable)
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Scale the training and test data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

#scaler=MinMaxScaler()
model= load_model('CardioVascular.h5' )

app = Flask(__name__)
app.secret_key="123"

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = scaler.transform([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        data = data.reshape((data.shape[0], data.shape[1], 1))
        #print(data)
        my_prediction = model.predict(data)
        #print(my_prediction)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
        app.run()

