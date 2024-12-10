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
    if tree[0] == "Leaf":
        return tree[1]  # Return the class label stored in the leaf
    # Tree is structured as ["Attribute", attribute_name, ...]
    attribute_index = header.index(tree[1])
    attribute_value = instance[attribute_index]

    # Now we need to traverse based on the attribute's value
    for subtree in tree[2:]:  # Iterate through the subtrees
        if subtree[0] == "Value" and subtree[1] == attribute_value:
            return tdidt_predict(
                instance, subtree[2], header
            )  # Recurse into the next subtree
    # If no match was found (shouldn't happen in a well-formed tree), we can return a default or raise an error
    raise ValueError(
        f"Unrecognized value '{attribute_value}' for attribute '{tree[1]}'"
    )


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
