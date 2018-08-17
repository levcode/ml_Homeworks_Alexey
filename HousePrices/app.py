import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn import tree
from flask import Flask
from flask import request
app = Flask(__name__)

f = open('model1.pickle','rb')
mod1 = pickle.load(f)
f.close()
f = open('model2.pickle','rb')
mod2 = pickle.load(f)
f.close()
@app.route("/")
def hello():
    return "Request example: /predict?full_sq=45&life_sq=30&floor=3&max_floor=4"

@app.route("/predict")
def pred():
    full_sq = float(request.args.get('full_sq'))
    life_sq = float(request.args.get( 'life_sq'))
    floor = float(request.args.get( 'floor'))
    max_floor = float(request.args.get('max_floor'))
    X = np.array([full_sq,life_sq,floor,max_floor])
    pred1 = mod1.predict(X.reshape(1,-1))
    pred2 = mod2.predict(X.reshape(1,-1))
    pred_mean = (pred1[0]+pred2[0])/2
    return "Price is {0}".format(pred_mean)


 
if __name__ == "__main__":
    app.run()