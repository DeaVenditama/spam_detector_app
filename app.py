import json
import re
import pickle
import numpy
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

bow_transformer = pickle.load(open("bow_transformer.pkl", "rb"))
tfidf_transformer = pickle.load(open("tfidf_transformer.pkl", "rb"))
spam_detect_model = pickle.load(open("spam_detect_model.pkl", "rb"))
ps = PorterStemmer()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict')
@cross_origin()
def predict():
    text = request.args.get("text")
    clean_text = preprocess_text(text)
    bow_check = bow_transformer.transform([clean_text])
    tfidf_check = tfidf_transformer.transform(bow_check)
    result = spam_detect_model.predict(tfidf_check)[0]
    print(result)
    return json.dumps(numpy.int32(result), cls=MyEncoder)

def preprocess_text(message):
    message = message.lower()
    message = re.sub('[^A-Za-z0-9 ]+', '', message)
    list_message = []
    for word in message.split():
        if word not in stopwords.words('english'):
            temp_word = ps.stem(word)
            list_message.append(temp_word)
    return " ".join(list_message)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)