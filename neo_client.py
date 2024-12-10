import requests  # a library for making http requests
import json  # a library for working with json

# TODO: Determine url
url_string = "https://cpsc322-finalproject-0itg.onrender.com/predict?estimated_diameter_min=0&estimated_diameter_max=1&relative_velocity=2&miss_distance=1"

response = requests.get(url=url_string)

# first thing, check the response's status code
# https://developer
# print(response.status_code)
if response.status_code == 200:
    # Status == Okay
    json_object = json.loads(response.text)
    prediction = json_object["prediction"]
    print("Prediction:", prediction)
if response.status_code == 400:
    # Status == Not Okay
    print("Not okay")
