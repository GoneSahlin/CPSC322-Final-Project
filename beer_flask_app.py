import os
import pickle

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

from mysklearn.myclassifiers import MyDecisionTreeClassifier

app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def index_page():
    prediction = ""
    if request.method == "POST":
        level = request.form["level"]
        lang = request.form["lang"]
        tweets = request.form["tweets"]
        phd = request.form["phd"]
        prediction = predict_interviews_well([level, lang, tweets, phd])
    print("prediction:", prediction)
    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction)


@app.route('/predict', methods=["GET"])
def predict():
    level = request.args.get("level")
    lang = request.args.get("lang")
    tweets = request.args.get("tweets")
    phd = request.args.get("phd")

    prediction = predict_interviews_well([level, lang, tweets, phd])
    if prediction is not None:
        # success!
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400


def predict_interviews_well(unseen_instance):
    # deserialize to object (unpickle)
    infile = open("tree.p", "rb")
    dt_clf = pickle.load(infile)
    infile.close()
    try:
        return dt_clf.predict(unseen_instance)
    except:
        return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)