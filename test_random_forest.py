import numpy as np

from mysklearn.myclassifiers import MyRandomForestClassifier

header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

def test_fit():
    forest = MyRandomForestClassifier(5, 3, 2)
    forest.fit(X_train_interview, y_train_interview)
    
    # ensure enough trees were kept
    assert len(forest.classifiers) == 3

    # ensure each tree is unique
    tree_list = []
    for classifier in forest.classifiers:
        tree_list.append(classifier.tree)
    unique_trees = set(tree_list)
    assert len(unique_trees) == 3

def test_predict():
    pass