"""
Programmers: Michael D'Arcy-Evans and Isabel Tilles
Class: CPSC322-01, Fall 2024
Final Project
12/6/2024
We attempted the bonus.

Description: An example testing the implementation various classifier techniques and their accuracy
"""

import numpy as np
from tabulate import tabulate
from mysklearn import myutils


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                use numpy for your generator
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    myutils.set_random_seed(random_state)

    if shuffle:
        X, y = myutils.shuffle_data(X, y)

    n_test = (
        int(np.ceil(len(X) * test_size)) if isinstance(test_size, float) else test_size
    )
    X_train = X[:-n_test]
    X_test = X[-n_test:]
    y_train = y[:-n_test]
    y_test = y[-n_test:]

    return list(X_train), list(X_test), list(y_train), list(y_test)


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    myutils.set_random_seed(random_state)

    if shuffle:
        np.random.shuffle(indices)

    fold_sizes = myutils.calculate_fold_sizes(n_samples, n_splits)
    folds = []

    for fold in range(n_splits):
        test_indices = indices[fold_sizes[:fold].sum() : fold_sizes[: fold + 1].sum()]
        train_indices = np.concatenate(
            (
                indices[: fold_sizes[:fold].sum()],
                indices[fold_sizes[: fold + 1].sum() :],
            )
        )
        folds.append((list(train_indices), list(test_indices)))

    return folds


# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    myutils.set_random_seed(random_state)

    if shuffle:
        np.random.shuffle(indices)

    label_indices = {}
    for idx, label in zip(indices, y):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)

    folds = [[] for _ in range(n_splits)]
    for _, class_indices in label_indices.items():
        n_class_samples = len(class_indices)
        if shuffle:
            np.random.shuffle(class_indices)
        fold_sizes = myutils.calculate_fold_sizes(n_class_samples, n_splits)
        current = 0
        for fold in range(n_splits):
            start, stop = current, current + fold_sizes[fold]
            folds[fold].extend(class_indices[start:stop])
            current = stop
    final_folds = []
    for fold in range(n_splits):
        test_indices = folds[fold]
        train_indices = np.concatenate([folds[i] for i in range(n_splits) if i != fold])
        final_folds.append((list(train_indices), list(test_indices)))

    return final_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if n_samples is None:
        n_samples = len(X)
    myutils.set_random_seed(random_state)

    sample_indices = np.random.choice(len(X), size=n_samples, replace=True)
    X_sample = [X[i] for i in sample_indices]
    out_of_bag_indices = list(set(range(len(X))) - set(sample_indices))
    X_out_of_bag = [X[i] for i in out_of_bag_indices]
    y_sample = None
    y_out_of_bag = None

    if y is not None:
        y_sample = [y[i] for i in sample_indices]
        y_out_of_bag = [y[i] for i in out_of_bag_indices]

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    label_index = {label: index for index, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        if true in label_index and pred in label_index:
            true_index = label_index[true]
            pred_index = label_index[pred]
            matrix[true_index][pred_index] += 1

    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct_count = myutils.count_correct_predictions(y_true, y_pred)
    if normalize:
        return correct_count / len(y_true) if len(y_true) > 0 else 0.0
    return correct_count


def random_subsample(classifier, X, y, k=10, test_size=0.33, random_state=None):
    """Perform random sub-sampling of the dataset.

    Args:
        X (list of list of obj): The list of samples.
        y (list of obj): The target y values (parallel to X).
        k (int): The number of sub-samples.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        accuracies (list of float): List of accuracies for each sub-sample.
        error_rates (list of float): List of error rates for each sub-sample.
    """
    accuracies = []

    for _ in range(k):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions, normalize=True)
        accuracies.append(accuracy)
    accuracy = np.mean(accuracies)
    error_rate = 1 - accuracy
    return accuracy, error_rate


def cross_val_predict(classifier, X, y, n_splits, stratify=False):
    """Performs k-fold or stratified k-fold cross-validation and returns predictions for each sample.

    Args:
        classifier: The classifier to be trained and tested.
        X (list of list of obj): The list of samples.
        y (list of obj): The target y values (parallel to X).
        n_splits (int): Number of folds for cross-validation.
        stratify (bool): If True, use stratified k-fold cross-validation. If False, use regular k-fold.

    Returns:
        predictions (list of obj): The predictions for each sample.
    """
    if stratify:
        folds = stratified_kfold_split(X, y, n_splits)
    else:
        folds = kfold_split(X, n_splits)

    predictions = [None] * len(y)

    for train_indices, test_indices in folds:
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]

        classifier.fit(X_train, y_train)
        fold_predictions = classifier.predict(X_test)

        for idx, pred in zip(test_indices, fold_predictions):
            predictions[idx] = pred

    return predictions


def bootstrap_method(classifier, X, y, k=10, random_state=None):
    """Compute predictive accuracy and error rate using bootstrap sampling.

    Args:
        classifier: The classifier to be trained and tested.
        X (list of list of obj): The list of samples.
        y (list of obj): The target y values (parallel to X).
        k (int): The number of bootstrap samples.
        random_state (int): Random seed for reproducibility.

    Returns:
        accuracy (float): Average accuracy across all bootstrap samples.
        error_rate (float): Average error rate across all bootstrap samples.
    """
    accuracies = []

    for _ in range(k):
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = bootstrap_sample(
            X, y, random_state=random_state
        )

        if y_out_of_bag:  # Check if we have out-of-bag samples
            classifier.fit(X_sample, y_sample)
            predictions = classifier.predict(X_out_of_bag)
            accuracy = accuracy_score(y_out_of_bag, predictions, normalize=True)
            accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies) if accuracies else 0.0
    average_error_rate = 1 - average_accuracy

    return average_accuracy, average_error_rate


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # If no labels are provided, use unique values in y_true
    if labels is None:
        labels = list(np.unique(y_true))

    # Default pos_label is the first class in the labels
    if pos_label is None:
        pos_label = labels[0]

    # Step 1: Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels)

    # Step 2: Find index of pos_label in labels
    pos_label_index = labels.index(pos_label)

    # Step 3: Extract True Positives (TP) and False Positives (FP)
    true_positive = matrix[pos_label_index][
        pos_label_index
    ]  # True Positives (diagonal element)
    false_positive = sum(
        matrix[i][pos_label_index] for i in range(len(labels)) if i != pos_label_index
    )  # False Positives (sum of the column except the diagonal)

    # Step 4: Compute precision
    if true_positive + false_positive == 0:
        return 0.0  # If there are no positive predictions, precision is not defined, return 0.0

    precision = true_positive / (true_positive + false_positive)
    return precision


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # If no labels are provided, use unique values in y_true
    if labels is None:
        labels = list(np.unique(y_true))

    # Default pos_label is the first class in the labels
    if pos_label is None:
        pos_label = labels[0]

    # Step 1: Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels)

    # Step 2: Find index of pos_label in labels
    pos_label_index = labels.index(pos_label)

    # Step 3: Extract True Positives (TP) and False Negatives (FN)
    true_positive = matrix[pos_label_index][
        pos_label_index
    ]  # True Positives (diagonal element)
    false_negative = (
        sum(matrix[pos_label_index]) - true_positive
    )  # False Negatives (sum of the row except the diagonal)

    # Step 4: Compute recall
    if true_positive + false_negative == 0:
        return 0.0  # If there are no true positives and false negatives, recall is not defined, return 0.0

    recall = true_positive / (true_positive + false_negative)
    return recall


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(
        y_true, y_pred, labels=labels, pos_label=pos_label
    )
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    if precision + recall == 0:
        return 0.0  # If Precision + Recall = 0, F1 is undefined (or 0)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def classification_report(y_true, y_pred, labels=None, output_dict=False):
    """Build a text report and a dictionary showing the main classification metrics.

    Args:
        y_true(list of obj): The ground_truth target y values.
        y_pred(list of obj): The predicted target y values (parallel to y_true).
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true.
        output_dict(bool): If True, return output as dict instead of a str.

    Returns:
        report(str or dict): Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision':0.5, 'recall':1.0, 'f1-score':0.67, 'support':1},
                 'label 2': {...}, ...}
    """
    if labels is None:
        labels = list(set(y_true))  # Derive labels from y_true if not provided

    # Initialize the metrics for each class
    metrics = {}

    for label in labels:
        metrics[label] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": 0,
        }

    # Calculate confusion matrix and metrics for each class
    for label in labels:

        # Support is the number of occurrences of the label in the true values
        support = sum(1 for true in y_true if true == label)

        precision_value = binary_precision_score(y_true, y_pred, pos_label=label)
        recall_value = binary_recall_score(y_true, y_pred, pos_label=label)
        f1_value = binary_f1_score(y_true, y_pred, pos_label=label)

        # Store calculated metrics
        metrics[label]["precision"] = precision_value
        metrics[label]["recall"] = recall_value
        metrics[label]["f1-score"] = f1_value
        metrics[label]["support"] = support

    # Calculate average metrics (macro avg)
    total_support = sum(metrics[label]["support"] for label in labels)
    total_precision = sum(metrics[label]["precision"] for label in labels)
    total_recall = sum(metrics[label]["recall"] for label in labels)
    total_f1 = sum(metrics[label]["f1-score"] for label in labels)
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    acc_metrics = {
        "precision": None,
        "recall": None,
        "f1-score": accuracy,
        "support": total_support,
    }
    metrics["accuracy"] = acc_metrics
    # Macro Average
    macro_precision = total_precision / len(labels) if len(labels) > 0 else 0.0
    macro_recall = total_recall / len(labels) if len(labels) > 0 else 0.0
    macro_f1 = total_f1 / len(labels) if len(labels) > 0 else 0.0

    avg_metrics = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": total_support,
    }

    # Add the averages to the metrics
    metrics["macro avg"] = avg_metrics

    # Weighted
    total_precision_weighted = sum(
        metrics[label]["support"] * metrics[label]["precision"] for label in labels
    )
    total_recall_weighted = sum(
        metrics[label]["support"] * metrics[label]["recall"] for label in labels
    )
    total_f1_weighted = sum(
        metrics[label]["support"] * metrics[label]["f1-score"] for label in labels
    )

    weighted_precision = (
        total_precision_weighted / total_support if total_support > 0 else 0.0
    )
    weighted_recall = (
        total_recall_weighted / total_support if total_support > 0 else 0.0
    )
    weighted_f1 = total_f1_weighted / total_support if total_support > 0 else 0.0

    # Add the weighted averages to the metrics dictionary
    weighted_avg_metrics = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": total_support,
    }

    metrics["weighted avg"] = weighted_avg_metrics

    if output_dict:
        return metrics

    # Create a nicely formatted table
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    table = []
    for label, data in metrics.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            table.append(
                [
                    label,
                    f"{data['precision']:.4f}",
                    f"{data['recall']:.4f}",
                    f"{data['f1-score']:.4f}",
                    data["support"],
                ]
            )

    # Add the averages row to the table
    table.append(
        [
            "accuracy",
            acc_metrics["precision"],
            acc_metrics["recall"],
            f"{acc_metrics['f1-score']:.4f}",
            acc_metrics["support"],
        ]
    )
    table.append(
        [
            "macro avg",
            f"{avg_metrics['precision']:.4f}",
            f"{avg_metrics['recall']:.4f}",
            f"{avg_metrics['f1-score']:.4f}",
            avg_metrics["support"],
        ]
    )
    table.append(
        [
            "weighted avg",
            f"{weighted_avg_metrics['precision']:.4f}",
            f"{weighted_avg_metrics['recall']:.4f}",
            f"{weighted_avg_metrics['f1-score']:.4f}",
            weighted_avg_metrics["support"],
        ]
    )

    return tabulate(table, headers=headers, tablefmt="grid")


def pseudo_classification_report(y_true, y_pred, labels=None,classifier_name = None):
    headers = ["Is Hazardous:"] + labels
    if classifier_name is None:
        classifier_name = "Decision Tree"
    if labels is None:
        labels = list(set(y_true))
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    precision = binary_precision_score(y_true, y_pred, labels=labels)
    recall = binary_recall_score(y_true, y_pred, labels=labels)
    f1 = binary_f1_score(y_true, y_pred, labels=labels)
    matrix = confusion_matrix(y_true, y_pred, labels)
    return f"{classifier_name} Classifier: Accuracy = {accuracy:.2f}, Error Rate = {error_rate:.2f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}\n{tabulate(matrix, headers=headers, showindex=labels, tablefmt="fancy_grid")}"
