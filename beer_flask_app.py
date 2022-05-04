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
        beer_style = request.form["style"]
        abv = request.form["abv"]
        brewery_rating = request.form["brewery_rating"]
        brewery_country = request.form["brewery_country"]
        prediction = predict_interviews_well([beer_style, abv, brewery_rating, brewery_country])
    print("prediction:", prediction)
    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction)


@app.route('/predict', methods=["GET"])
def predict():
    beer_style = request.args.get("style")
    abv = request.args.get("abv")
    brewery_rating = request.args.get("brewery rating")
    brewery_country = request.args.get("brewery country")

    prediction = predict_interviews_well([beer_style, abv, brewery_rating, brewery_country])
    if prediction is not None:
        # success!
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400


def predict_interviews_well(unseen_instance):
    # deserialize to object (unpickle)
    infile = open("classifier.p", "rb")
    clf = pickle.load(infile)
    infile.close()
    try:
        unseen_instance[1] = float(unseen_instance[1])
        unseen_instance[3] = float(unseen_instance[3])

        return clf.predict([unseen_instance])
        # return unseen_instance
    except:
        return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)