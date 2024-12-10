import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)


def load_model():
    # unpickle header and tree
    infile = open("pickled_example.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    return header, tree


def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]  # label
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(header, value_list[2], instance)


# We need to add "routes" := a function that handles a request
# e.g. the HTML content for a home page, json response for a predict API endpoint
@app.route("/")
def index():
    # return content and status code
    return "<h1>Welcome to the interview predictor app</h1>", 200


@app.route("/predict")
def predict():
    # parse an unseen instance from the query string -- in the request object
    estimated_diameter_min = request.args.get(
        "estimated_diameter_min"
    )  # default to None if no level
    estimated_diameter_max = request.args.get("estimated_diameter_max")
    relative_velocity = request.args.get("relative_velocity")
    miss_distance = request.args.get("miss_distance")
    instance = [
        estimated_diameter_min,
        estimated_diameter_max,
        relative_velocity,
        miss_distance,
    ]
    header, tree = load_model()
    pred = tdidt_predict(header, tree, instance)
    if pred is not None:
        return jsonify({"Prediction": pred}), 200
    return "Error making a prediction", 400


if __name__ == "__main__":
    # header, tree = load_model()
    # print(header)
    # print(tree)
    app.run(host="0.0.0.0", port=5000, debug=False)
    # TODO: when deploying app to production, set debug = False
    # Check host / port values
