"""
Programmers: Michael D'Arcy-Evans and Isabel Tilles
Class: CPSC322-01, Fall 2024
Final Project
12/6/2024
We attempted the bonus.

Description: numerous classifiers used to predict labels based on attribute values
"""

import numpy as np
from graphviz import Graph
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn import myevaluation


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor if regressor else MySimpleLinearRegressor()

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        numeric_predictions = self.regressor.predict(X_test)
        return [self.discretizer(pred) for pred in numeric_predictions]


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for test_point in X_test:
            dists = [
                myutils.compute_euclidean_distance(test_point, train_point)
                for train_point in self.X_train
            ]
            # Get the indices of the k smallest distances
            k_indices = np.argsort(dists)[: self.n_neighbors]
            distances.append([dists[idx] for idx in k_indices])
            neighbor_indices.append(k_indices.tolist())
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        indices = self.kneighbors(X_test)[1]
        y_predicted = []
        for i in range(len(X_test)):
            # Gather the labels of the k nearest neighbors
            k_nearest_labels = [self.y_train[idx] for idx in indices[i]]
            # Count the occurrences of each label
            label_counts = myutils.count_label_occurrences(k_nearest_labels)

            # Find the label with the maximum count
            most_common_label = max(label_counts, key=label_counts.get)
            y_predicted.append(most_common_label)

        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self, strategy="most_frequent"):
        """Initializer for DummyClassifier."""
        self.strategy = strategy
        self.most_common_label = None
        self.class_probabilities = None
        self.class_labels = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        _x_train = X_train
        label_counts = myutils.count_label_occurrences(y_train)
        self.class_labels = list(label_counts.keys())

        if self.strategy == "most_frequent":
            self.most_common_label = max(label_counts, key=label_counts.get)
        elif self.strategy == "stratified":
            total_count = len(y_train)
            self.class_probabilities = [
                count / total_count for count in label_counts.values()
            ]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.strategy == "most_frequent":
            if self.most_common_label is None:
                raise ValueError("The model has not been fitted yet.")
            return [self.most_common_label] * len(X_test)
        if self.strategy == "stratified":
            if self.class_probabilities is None:
                raise ValueError("The model has not been fitted yet.")
            return np.random.choice(
                self.class_labels, size=len(X_test), p=self.class_probabilities
            )
        raise ValueError(f"Unknown strategy: {self.strategy}")


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier."""
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # Step 1: Count the occurrences of each class in y_train
        class_counts = {}
        total_samples = len(y_train)

        for label in y_train:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Step 2: Compute the prior probabilities
        self.priors = {}
        for class_label, count in class_counts.items():
            self.priors[class_label] = [
                count,
                total_samples,
            ]  # Store as numerator/denominator

        # Step 3: Initialize feature_counts structure for each class and attribute
        feature_counts = {}

        # Loop over each sample and its corresponding label
        for i, sample in enumerate(X_train):
            label = y_train[i]  # Get the label for the current sample

            # Initialize feature_counts for each class if not already initialized
            if label not in feature_counts:
                feature_counts[label] = {}

            # Loop over the features in the sample
            for feature_index, feature_value in enumerate(sample):
                # Initialize the feature index (i.e., attribute) in the dictionary if not present
                if feature_index not in feature_counts[label]:
                    feature_counts[label][feature_index] = {}

                # Initialize the feature value in the dictionary if it's not present
                if feature_value not in feature_counts[label][feature_index]:
                    feature_counts[label][feature_index][feature_value] = [
                        0,
                        0,
                    ]  # [count, total]

                # Increment the count of this feature value for the given class and feature index
                feature_counts[label][feature_index][feature_value][
                    1
                ] += 1  # Increment the total for this value
                if label == y_train[i]:  # Increment the count if the label matches
                    feature_counts[label][feature_index][feature_value][
                        0
                    ] += 1  # Increment the count for this value

        # Step 4: Compute the posterior probabilities (feature value probabilities) for each feature given the class
        self.posteriors = {}
        for class_label, feature_dict in feature_counts.items():
            self.posteriors[class_label] = {}
            total_class_samples = class_counts[
                class_label
            ]  # Total number of samples for the class

            # Calculate posterior probabilities for each feature value given the class
            for feature_index, value_counts in feature_dict.items():
                for feature_value, count_total in value_counts.items():
                    count, _ = count_total
                    if feature_index not in self.posteriors[class_label]:
                        self.posteriors[class_label][feature_index] = {}
                    self.posteriors[class_label][feature_index][feature_value] = [
                        count,
                        total_class_samples,
                    ]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if not isinstance(X_test[0], list):
            X_test = [X_test]

        y_predicted = []  # This will store the predicted labels for all samples
        class_scores = {}  # Dictionary to store the score for each class

        for sample in X_test:
            # For each class
            for class_label, class_prior in self.priors.items():
                # Start with the prior for this class
                numerator = class_prior[0]
                denominator = class_prior[1]
                # Loop over the features in the sample and update the score
                for feature_index, feature_value in enumerate(sample):
                    # Check if the feature value exists in the posteriors for the given class and feature
                    if feature_value in self.posteriors[class_label][feature_index]:
                        # Get the posterior (numerator/denominator) for the given feature value
                        feature_prob = self.posteriors[class_label][feature_index][
                            feature_value
                        ]

                        # Multiply the numerators and denominators
                        numerator *= feature_prob[0]
                        denominator *= feature_prob[1]
                    else:
                        # If the feature value has not been seen during training, assume it does not exist
                        numerator *= 0

                # Store the computed numerator/denominator for this class
                class_scores[class_label] = (numerator, denominator)

            # Compare the products of numerators/denominators and pick the class with the highest ratio
            best_class = max(
                class_scores, key=lambda x: class_scores[x][0] / class_scores[x][1]
            )
            y_predicted.append(best_class)

        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        F(int): The number of attributes to select from when doing TDIDT algorithm (only used if part of a random forest)

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, F = None):
        """Initializer for MyDecisionTreeClassifier."""
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.F = F

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        header, attribute_domains = myutils.build_header_and_domains(X_train)
        available_attributes = header.copy()
        self.tree = myutils.tdidt(
            train_data, available_attributes, header, attribute_domains, F = self.F
        )

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        header, _ = myutils.build_header_and_domains(self.X_train)
        for instance in X_test:
            # Start from the root of the tree and traverse down
            prediction = self._predict_single(instance, self.tree, header)
            y_predicted.append(prediction)

        return y_predicted

    def _predict_single(self, instance, tree, header):
        """Helper function to predict the class for a single instance by traversing the tree.

        Args:
            instance(list): A single test instance (one row from X_test)
            tree(list): The decision tree structure

        Returns:
            str: The predicted class label
        """
        if tree[0] == "Leaf":
            return tree[1]  # Return the class label stored in the leaf
        # Tree is structured as ["Attribute", attribute_name, ...]
        attribute_index = header.index(tree[1])
        attribute_value = instance[attribute_index]

        # Now we need to traverse based on the attribute's value
        for subtree in tree[2:]:  # Iterate through the subtrees
            if subtree[0] == "Value" and subtree[1] == attribute_value:
                return self._predict_single(
                    instance, subtree[2], header
                )  # Recurse into the next subtree
        # If no match was found (shouldn't happen in a well-formed tree), we can return a default or raise an error
        raise ValueError(
            f"Unrecognized value '{attribute_value}' for attribute '{tree[1]}'"
        )

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        myutils.traverse_tree_print(self.tree, [], attribute_names, class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        dot_graph = Graph(
            strict=True,
            engine="dot",
            format="pdf",
            directory="./tree_vis",
        )
        # Start the recursive function to generate the nodes and edges
        myutils.add_nodes_and_edges(
            self.tree, parent=None, graph=dot_graph, attribute_names=attribute_names
        )
        dot_graph.save(dot_fname)
        dot_graph.render(pdf_fname, cleanup=True)

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        classifiers(list of obj): model containing the trees
        N (int): number of trees to generate for the forest initially
        M (int): number of 'best' trees to keep in the forest
        F (int): number of attributes to have each tree in the Forest consider

    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, N, M, F):
        """Initializer for MyRandomForestClassifier."""
        self.X_train = None
        self.y_train = None
        self.classifiers = []
        self.N = N
        self.M = M
        self.F = F

    def fit(self, X_train, y_train):
        """Fits a random forest classifier, forms all trees using X_train and y_train

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            
        """
        self.X_train = X_train
        self.y_train = y_train
        myutils.set_random_seed(0)

        self.classifiers = self.generate_final_forest()
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test based on majority voting in the forest

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for X in X_test:
            result = myutils.calculate_majority_votes(self.classifiers, X)
            y_predicted.append(result)

        return y_predicted

    def generate_initial_forest(self, random_state = None):
        """ 
        Generate N trees
        """
        N_size_forest = []
        for _ in range(self.N):
            # build a tree
            tree_classifier = MyDecisionTreeClassifier(self.F)
            X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(self.X_train, self.y_train, random_state = random_state)
            tree_classifier.fit(X_sample, y_sample)
            full_tree = (tree_classifier, X_out_of_bag, y_out_of_bag)
            N_size_forest.append(full_tree)
        return N_size_forest

    def refine_forest(self, N_size_forest):
        """Select top M trees based on Recall

        Args:
            N_size_forest (list of lists): initial trees & their info generated for random forest
                Each tree list looks like - [tree classifier, X_test, y_test]
            M (int): size of final forest

        Returns:
            M_size_forest (list of obj): final forest of size M; list consists of classifiers
            
        Note:
            This tests for recall because that is what we want to focus on for our NEO Classifier

        """
        M_size_forest = []
        tree_recall_dict = dict.fromkeys(range(len(N_size_forest)), None)
        for index, tree in enumerate(N_size_forest):
            # have to unpack tree
            classifier = tree[0]
            X_test = tree[1]
            y_test = tree[2]

            y_predicted = classifier.predict(X_test)
            recall_score = myevaluation.binary_recall_score(y_test, y_predicted)
            tree_recall_dict[index] = recall_score
        # get indexes of top M trees with best recall, to keep
        top_M_indexes = sorted(tree_recall_dict, key=tree_recall_dict.get, reverse=True)[:self.M]
        for index in top_M_indexes:
            # add the best trees to return list
            M_size_forest.append(N_size_forest[index][0])
        return M_size_forest

    def generate_final_forest(self, random_state = None):
        # make N trees
        initial_forest = self.generate_initial_forest(random_state)
        # pare it down to M trees
        final_forest = self.refine_forest(initial_forest)
        return final_forest