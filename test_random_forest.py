import numpy as np

from mysklearn.myclassifiers import MyRandomForestClassifier
import mysklearn.myutils as myutils

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

fitted_forest = [
    # tree 1
    ["Attribute", "att0",
        ["Value", "Junior", 
            ["Leaf", "True", 9, 14]
         ],
        ["Value", "Mid", 
            ["Leaf", "True", 1, 14]
        ],
        ["Value", "Senior",
            ["Attribute", "att2",
                ["Value", "no", 
                    ["Leaf", "False", 2, 4]
                ],
                ["Value", "yes", 
                    ["Leaf", "True", 2, 4]
                ]
            ]
        ]
    ],
    # tree 2
    ["Attribute", "att1",
        ["Value", "Java", 
            ["Leaf", "False", 3, 14]
        ],
        ["Value", "Python",
            ["Attribute", "att2",
                ["Value", "no", 
                    ["Leaf", "False", 2, 7]
                ],
                ["Value", "yes", 
                    ["Leaf", "True", 5, 7]
                ]
            ]
        ],
        ["Value", "R", 
            ["Leaf", "True", 4, 14]
        ]
    ],
    # tree 3
    ["Attribute", "att1",
        ["Value", "Java", 
            ["Leaf", "False", 2, 14]
        ],
        ["Value", "Python",
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Leaf", "True", 4, 8]
                ],
                ["Value", "Mid", 
                    ["Leaf", "True", 2, 8]
                ],
                ["Value", "Senior", 
                    ["Leaf", "False", 2, 8]
                ]
            ]
        ],
        ["Value", "R", 
            ["Leaf", "False", 4, 4]
        ]
    ]
]

X_test_1 = ["Senior", "Python", "yes", "no"] # True
X_test_2 = ["Senior", "Python", "no", "no"] # False
X_test_3 = ["Junior", "Java", "no", "yes"] # False



def test_fit():
    forest = MyRandomForestClassifier(5, 3, 2)
    forest.fit(X_train_interview, y_train_interview)
    
    # ensure enough trees were kept
    assert len(forest.classifiers) == 3

    # ensure each tree is unique
    tree_list = []
    for classifier in forest.classifiers:
        tree_list.append(classifier.tree)
    print(tree_list)

    unique_trees = myutils.check_unique_trees(tree_list)
    assert unique_trees == True


def test_predict():
    forest = MyRandomForestClassifier(5, 3, 2)
    forest.fit(X_train_interview, y_train_interview)

    y_pred = forest.predict([X_test_1, X_test_2, X_test_3])

    y_expected = ["True", "False", "False"]

    assert y_pred == y_expected