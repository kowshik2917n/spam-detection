from flask import Flask,render_template,request,jsonify
import pickle
from pymongo import MongoClient
from datetime import datetime

app=Flask(__name__)

client=MongoClient("mongodb://127.0.0.1:27017/")
db=client["spamDB"]
collection=db["messages"]

model=pickle.load(open("model/spam_model.pkl","rb"))
vectorizer=pickle.load(open("model/vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    msg=request.json["message"]
    data=vectorizer.transform([msg])
    result=model.predict(data)[0]

    collection.insert_one({
        "message":msg,
        "result":result,
        "date":datetime.now()
    })

    return jsonify(result)

app.run(debug=True)
