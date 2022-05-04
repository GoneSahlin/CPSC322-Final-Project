import requests
import json

def request_prediction(instance):
    url = "http://localhost:5000/predict?"
    url += "level=" + instance[0]
    url += "&lang=" + instance[1]
    url += "&tweets=" + instance[2]
    url += "&phd=" + instance[3]

    # open the URL and read the server's response
    response = requests.get(url=url)
    json_object = json.loads(response.text)
    print("response:", json_object)
    return json_object["prediction"]

X_test = [["Junior", "Java", "yes", "no"],
          ["Junior", "Java", "yes", "yes"]]
for unseen_instance in X_test:
    prediction = request_prediction(unseen_instance)
    print("prediction for", unseen_instance, ":", prediction)