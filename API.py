from flask import Flask, request
import joblib

app = Flask(__name__)
classifier = joblib.load('clasificator.pkl')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json 
    features = data['features'] 

    prediction = classifier.predict([features])[0]

    return {'prediction': prediction}

if __name__ == '__main__':
    app.run()
