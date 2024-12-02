"""
Programmer: Michael D'Arcy-Evans
Class: CPSC322-01, Fall 2024
Programming Assignment #6
11/5/2024
I attempted the bonus.

Description: Reused utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
from graphviz import Graph
import mysklearn.myevaluation as myevaluation


def discretizer_function(value):
    """Map a numeric value to a categorical label ("high" or "low").

    Args:
        value(float): The numeric value to be categorized.

    Returns:
        str: "high" if value is greater than or equal to 100, "low" otherwise.
    """
    return "high" if value >= 100 else "low"


def set_random_seed(random_state):
    """Set the random seed for reproducibility.

    Args:
        random_state(int): Integer used for seeding a random number generator for reproducible results.
            Use random_state to seed your random number generator; use numpy for your generator.
    """
    if random_state is not None:
        np.random.seed(random_state)


def shuffle_data(X, y):
    """Shuffle data and maintain parallel order while ensuring minimal element position overlap.

    Args:
        X (list of list of obj): The list of samples.
            The shape of X is (n_samples, n_features).
        y (list of obj): The target y values (parallel to X).
            The shape of y is n_samples.

    Returns:
        tuple: A tuple containing the shuffled X and y, maintaining their parallel order.

    Notes:
        This function randomly shuffles the input lists X and y, maintaining their parallel
        relationship, and ensures that no more than one element remains in the same position
        after shuffling.
    """
    original_x = X.copy()
    same_position_count = len(X)

    while same_position_count >= 2:
        combined = list(zip(X, y))
        np.random.shuffle(combined)
        X, y = zip(*combined)
        X = [list(item) for item in X]
        y = list(y)
        same_position_count = sum(
            1 for orig, shuffled in zip(original_x, X) if orig == shuffled
        )

    return X, y


def calculate_fold_sizes(n_samples, n_splits):
    """Calculate the sizes of each fold for cross-validation.

    Args:
        n_samples(int): The total number of samples.
        n_splits(int): The number of folds.

    Returns:
        An array containing the sizes of each fold.
    """
    fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    return fold_sizes


def normalize(data):
    """Normalize specified attributes (cylinders, weight, acceleration) from the dataset.

    Args:
        data(list of list of float): The dataset where each inner list contains
            the attributes [cylinder, weight, acceleration, ...].

    Returns:
        list of list of float: The dataset with normalized cylinders, weights, and accelerations.
    """
    cylinders = [x[0] for x in data]
    weights = [x[1] for x in data]
    accelerations = [x[2] for x in data]

    normalized_cylinders = normalize_helper(cylinders)
    normalized_weights = normalize_helper(weights)
    normalized_accelerations = normalize_helper(accelerations)

    normalized_dataset = [
        [normalized_cylinders[i], normalized_weights[i], normalized_accelerations[i]]
        for i in range(len(data))
    ]
    return normalized_dataset


def normalize_helper(values):
    """Normalize a list of numeric values to the range [0, 1].

    Args:
        values(list of float): The list of numeric values to normalize.

    Returns:
        list of float: The normalized values in the range [0, 1].
    """
    min_value = min(values)
    max_value = max(values)
    return [(value - min_value) / (max_value - min_value) for value in values]


def compute_euclidean_distance(v1, v2):
    """Calculate the Euclidean distance between two vectors.

    Args:
        v1(list of float): The first vector.
        v2(list of float): The second vector.

    Returns:
        float: The Euclidean distance between v1 and v2.
    """
    distance = 0

    for i, _ in enumerate(v1):
        if isinstance(v1[i], (int, float)) and isinstance(v2[i], (int, float)):
            # Continuous features: Calculate the squared difference for Euclidean distance
            distance += (v1[i] - v2[i]) ** 2
        else:
            # Nominal features: Use 1 if different, 0 if the same
            distance += 1 if v1[i] != v2[i] else 0

    # Return the square root of the summed squared differences (Euclidean distance)
    return np.sqrt(distance)


def count_correct_predictions(y_true, y_pred):
    """Count the number of correct predictions.

    Args:
        y_true(list of obj): The ground truth target values.
            The shape of y_true is n_samples.
        y_pred(list of obj): The predicted target values (parallel to y_true).
            The shape of y_pred is n_samples.

    Returns:
        int: The count of correctly predicted samples.
    """
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred)


def count_label_occurrences(labels):
    """Counts occurrences of each label in the provided list.

    Args:
        labels(list): A list of labels.

    Returns:
        dict: A dictionary with labels as keys and their counts as values.
    """
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    return label_counts


def log2(x):
    """
    Calculate the base-2 logarithm of a number. If the input is less than or equal to 0,
    returns 0 to handle edge cases.

    Args:
        x (float): The number to compute the base-2 logarithm of.

    Returns:
        float: The base-2 logarithm of x, or 0 if x <= 0.
    """
    if x <= 0:
        return 0  # Handle edge case (log2 is undefined for x <= 0)
    return np.log(x) / np.log(2)


def entropy(instances):
    """
    Calculate the entropy of a set of instances, which is a measure of the uncertainty in the class labels.

    Args:
        instances (list of list): The dataset, where each element is an instance (a list) and the last element
                                   of each instance is the class label.

    Returns:
        float: The entropy of the dataset.
    """
    # Extract the class labels (assumes class label is the last element in each instance)
    labels = [instance[-1] for instance in instances]
    label_counts = count_label_occurrences(labels)
    total = len(instances)
    # Calculate entropy: - sum(p_i * log2(p_i))
    entropy_value = 0
    for count in label_counts.values():
        p_i = count / total
        entropy_value -= p_i * log2(p_i)  # log2 calculation is handled separately

    return entropy_value


def information_gain(instances, partitions):
    """
    Calculate the information gain for a given set of partitions of instances. Information gain measures the
    reduction in entropy due to splitting the data on a particular attribute.

    Args:
        instances (list of list): The dataset, where each element is an instance (a list) and the last element
                                   of each instance is the class label.
        partitions (dict): A dictionary where the keys are attribute values and the values are lists of instances
                           corresponding to each partition.

    Returns:
        float: The information gain for the given partitions.
    """
    # Calculate the entropy of the full dataset (S)
    base_entropy = entropy(instances)

    # Calculate weighted entropy of the partitions
    total_instances = len(instances)
    weighted_entropy = 0
    for partition in partitions.values():
        partition_size = len(partition)
        if partition_size == 0:
            continue  # Skip empty partitions
        # Calculate the weight of the partition
        weight = partition_size / total_instances
        # Calculate the entropy of the partition
        partition_entropy = entropy(partition)
        weighted_entropy += weight * partition_entropy
    return base_entropy - weighted_entropy


def majority_class(instances):
    """
    Return the class label that occurs most frequently in the given set of instances. If there is a tie,
    the alphabetically first class is returned.

    Args:
        instances (list of list): The dataset, where each element is an instance (a list) and the last element
                                   of each instance is the class label.

    Returns:
        str: The majority class label.
    """
    labels = [instance[-1] for instance in instances]
    class_counts = count_label_occurrences(labels)

    # Find the class with the maximum occurrences
    max_count = max(class_counts.values())

    # Get all classes that have the max count
    majority_classes = [
        label for label, count in class_counts.items() if count == max_count
    ]

    # If there's a tie, return the alphabetically first one
    return min(majority_classes)


def partition_instances(instances, attribute, header, attribute_domains):
    """
    Partition the instances based on a specific attribute. Each partition corresponds to one of the possible
    values of the given attribute.

    Args:
        instances (list of list): The dataset, where each element is an instance (a list) and the last element
                                   of each instance is the class label.
        attribute (str): The attribute to partition the instances by (e.g., "att0", "att1", ...).
        header (list of str): The list of attribute names.
        attribute_domains (dict): A dictionary mapping attributes to their possible values.

    Returns:
        dict: A dictionary where the keys are attribute values and the values are lists of instances that match
              each attribute value.
    """
    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions


def all_same_class(instances):
    """
    Check if all instances in the dataset belong to the same class label.

    Args:
        instances (list of list): The dataset, where each element is an instance (a list) and the last element
                                   of each instance is the class label.

    Returns:
        bool: True if all instances have the same class label, False otherwise.
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    return True


def select_attribute(instances, available_attributes, header, attribute_domains):
    """
    Select the best attribute to split on, based on information gain.

    Args:
        instances (list): The list of instances (each instance is a list of attribute values).
        available_attributes (list): The list of attributes that can be used for splitting.
        header (list): The list of attribute names (e.g., ['att0', 'att1', ...]).
        attribute_domains (dict): A dictionary mapping each attribute to its possible values.

    Returns:
        str: The attribute with the highest information gain.
    """
    best_attribute = None
    best_gain = -1  # We want to maximize information gain, so start with a low value
    # Iterate through each available attribute
    for attribute in available_attributes:
        # Partition the data by the current attribute
        partitions = partition_instances(
            instances, attribute, header, attribute_domains
        )
        # Calculate information gain for this attribute
        gain = information_gain(instances, partitions)
        # Keep track of the best attribute with the highest information gain
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    return best_attribute


def tdidt(
    current_instances,
    available_attributes,
    header,
    attribute_domain,
    parent_node_size=None,
):
    """
    Recursively build a decision tree using the TDIDT (Top-Down Induction of Decision Trees) algorithm.
    The tree is built by selecting the best attribute to split on at each step, partitioning the instances,
    and creating subtrees for each partition.

    Args:
        current_instances (list of list): The current subset of instances to process.
        available_attributes (list of str): The list of attributes available for splitting.
        header (list of str): The list of attribute names.
        attribute_domain (dict): A dictionary mapping each attribute to its possible values.

    Returns:
        list: A nested list representing the decision tree. Each node is represented as a list where the first element
              is the node type (e.g., "Attribute", "Value", "Leaf"), and subsequent elements represent the node's
              attributes or subtrees.
    """
    # Select the best attribute based on information gain
    split_attribute = select_attribute(
        current_instances, available_attributes, header, attribute_domain
    )
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]

    # Partition the instances based on the selected attribute
    partitions = partition_instances(
        current_instances, split_attribute, header, attribute_domain
    )

    # Iterate through each partition
    for att_value in sorted(partitions.keys()):
        att_partition = partitions[att_value]
        value_subtree = ["Value", att_value]
        # Base Case 1: All instances in the partition have the same class
        if len(att_partition) > 0 and all_same_class(att_partition):
            # Add a leaf node with the majority class label and count of total instances in this partition
            leaf_class = majority_class(att_partition)  # Majority class label
            value_subtree.append(
                ["Leaf", leaf_class, len(att_partition), len(current_instances)]
            )

        # Base Case 2: No attributes left to split on (clash)
        elif len(att_partition) > 0 and not available_attributes:
            # Add a leaf node with the majority class label and count of total instances in the partition
            leaf_class = majority_class(current_instances)
            value_subtree.append(
                ["Leaf", leaf_class, len(att_partition), len(current_instances)]
            )

        # Base Case 3: Empty partition (empty partition)
        elif len(att_partition) == 0:
            # If no attributes left, replace the empty partition with a majority class leaf node
            majority_class_label = majority_class(current_instances)
            return [
                "Leaf",
                majority_class_label,
                len(current_instances),
                parent_node_size,
            ]

        # Recursive case: Split further by calling tdidt on the partition
        else:
            parent_node_size = len(current_instances)
            subtree = tdidt(
                att_partition,
                available_attributes.copy(),
                header,
                attribute_domain,
                parent_node_size,
            )
            value_subtree.append(subtree)

        # Append the value subtree to the main tree
        tree.append(value_subtree)

    return tree


def build_header_and_domains(X_train):
    """
    Build the header (attribute names) and the attribute domains (possible values for each attribute) from a dataset.

    Args:
        X_train (list of list): The dataset, where each element is an instance (a list of attribute values).

    Returns:
        tuple: A tuple containing:
            - header (list of str): The list of attribute names (e.g., ['att0', 'att1', ...]).
            - attribute_domains (dict): A dictionary mapping each attribute to its possible values.
    """
    # 1. Build header
    header = [f"att{i}" for i in range(len(X_train[0]))]
    # 2. Build attribute_domains
    attribute_domains = {}
    # Iterate through each attribute (column) in X_train
    for i in range(len(X_train[0])):  # Loop over the columns
        # Extract the values of the i-th attribute across all instances
        attribute_values = [instance[i] for instance in X_train]
        # Get unique values for this attribute (domain)
        attribute_domains[header[i]] = list(set(attribute_values))

    return header, attribute_domains


def traverse_tree_print(node, conditions, attribute_names, class_name):
    """
    Recursively traverse the decision tree and print the decision rules. The traversal starts at the root of the tree
    and follows the branches, printing conditions for each decision node and the classification at the leaf nodes.

    Args:
        node (list): The current node in the decision tree (either "Attribute", "Value", or "Leaf").
        conditions (list of str): A list of conditions leading up to the current node (used to print decision rules).
        attribute_names (list of str): A list of attribute names used in the decision tree.
        class_name (str): The name of the class attribute used for classification.

    Returns:
        None
    """
    # Leaf node (classification)
    if node[0] == "Leaf":
        label = node[1]  # This is the label (e.g., "yes" or "no")
        print("IF " + " AND ".join(conditions) + f" THEN {class_name} = {label}.")
        return

    # Attribute node (decision based on attribute)
    elif node[0] == "Attribute":
        attribute = node[1]  # This is the attribute (e.g., "att0", "att1", etc.)

        # Determine the attribute name (either "att0", "att1", ... or a custom name)
        if isinstance(attribute, str):  # "att0", "att1", etc.
            attribute_index = int(attribute[3:])  # Extract the index (0, 1, 2, ...)
            attribute_name = (
                attribute_names[attribute_index] if attribute_names else attribute
            )
        else:  # Handle direct integer attributes like 0, 1, 2
            attribute_name = (
                attribute_names[attribute] if attribute_names else f"att{attribute}"
            )
        # For each branch (value of the attribute), recurse into the subtree
        for branch in node[2:]:
            value = branch[1]  # This is the value (e.g., 1, 2, 3, "excellent", etc.)
            subtree = branch[2]  # This is the subtree for this value
            # Add the current condition to the list of conditions and recurse
            new_conditions = conditions + [f"{attribute_name} == {value}"]
            traverse_tree_print(subtree, new_conditions, attribute_names, class_name)


def add_nodes_and_edges(
    tree, parent=None, graph=None, node_counter=0, attribute_names=None, value=None
):
    """
    Recursively process the decision tree and add nodes/edges to a Graphviz object for visualization. Each node
    represents an attribute or leaf, and edges represent decisions based on attribute values.

    Args:
        tree (list): The decision tree to visualize, represented as a nested list.
        parent (str, optional): The parent node's name (default is None for the root node).
        graph (Graph, optional): A Graphviz Graph object to add nodes and edges to (default is None, creates a new graph).
        node_counter (int, optional): A counter to uniquely name each node (default is 0).
        attribute_names (list of str, optional): The list of attribute names (default is None).
        value (str, optional): The value of the attribute for the edge label (default is None).

    Returns:
        tuple: A tuple containing the name of the current node and the updated node counter.
    """
    # Initialize the graph if it's the first call
    if graph is None:
        graph = Graph(
            name="Decision Tree",
            strict=True,
            engine="dot",
            format="pdf",
            directory="./tree_vis",
        )

    # Extract the type of the current node ("Attribute", "Value", or "Leaf")
    node_type = tree[0]
    if node_type == "Attribute":
        node_name = f"node{node_counter}"
        node_counter += 1
        # Create a node for the attribute (e.g., "att0", "att1", etc.)
        node_label = tree[1]
        if attribute_names:
            index = int(node_label[3:])
            node_label = attribute_names[index]
        graph.node(node_name, label=node_label, shape="box")

        # If the node has a parent, create an edge from parent to this attribute node
        if parent is not None:
            graph.edge(parent, node_name, label=str(value))

        # Recursively process the child nodes (the values)
        values_node = tree[2:]  # The list of values to process
        for value_node in values_node:
            # The value that leads to the next node (this should be the edge label)
            child_node_name, node_counter = add_nodes_and_edges(
                value_node,
                parent=node_name,
                graph=graph,
                attribute_names=attribute_names,
                node_counter=node_counter,
            )

    elif node_type == "Value":
        next_node = tree[2]  # The next splitting node or leaf
        value = tree[1]
        child_node_name, node_counter = add_nodes_and_edges(
            next_node,
            parent=parent,
            graph=graph,
            node_counter=node_counter,
            attribute_names=attribute_names,
            value=value,
        )
        return child_node_name, node_counter

    elif node_type == "Leaf":
        node_name = f"node{node_counter}"
        node_counter += 1
        # Create a leaf node (final decision)
        leaf_node_name = f"node{node_counter}"
        node_counter += 1

        graph.node(
            leaf_node_name, label=f"{tree[1]}\n{tree[2]}/{tree[3]}", shape="ellipse"
        )

        # If the leaf has a parent, create an edge from parent to this leaf node
        if parent is not None:
            graph.edge(parent, leaf_node_name, label=str(value))  # The value

    return node_name, node_counter


def discretization(data, num_bins=None):
    """Groups data into equal-frequency bins and returns the frequency distribution.

    Args:
        data (list): A list of numeric values to be binned.
        num_bins (int): The number of bins to split the data into.

    Returns:
        bins (list): A list of tuples representing the bin ranges.
        bin_labels (list): A list of string labels for each bin.
    """
    if num_bins is None:
        num_bins = 3

    # Sort the data to facilitate equal-frequency binning
    sorted_data = np.sort(data)

    # Compute the bin edges using percentiles
    bin_edges = np.percentile(sorted_data, np.linspace(0, 100, num_bins + 1))

    # Create the bins by pairing up consecutive percentiles
    bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(num_bins)]

    # Label the bins with the range for each
    bin_labels = [f"{lower:.1f} to {upper:.1f}" for lower, upper in bins]

    # Count the frequency of values in each bin
    bin_counts = [0] * num_bins
    for value in sorted_data:
        for i, (lower, upper) in enumerate(bins):
            if lower <= value < upper:
                bin_counts[i] += 1
                break

    # Return the bins and labels
    return bins, bin_labels


def replace_outliers(data):
    if type(data[0]) is str:
        return data
    # if absolute value is greater than 3 or less than -3, it is an outlier
    z_threshold = 3.0
    mean = np.mean(data)
    std = np.std(data)
    # check if value is an outlier, and save its index
    outlier_indexes = [
        index for index, x in enumerate(data) if abs(x - mean) / std >= z_threshold
    ]
    valid_data = [
        value for index, value in enumerate(data) if index not in outlier_indexes
    ]
    median = np.median(valid_data)
    # if value is an outlier, replace it with the median. else, keep it the same
    new_data = [
        median if index in outlier_indexes else x for index, x in enumerate(data)
    ]
    return new_data


def preprocess_table(table):
    """Preprocesses a table by removing specified columns, shuffling data,
    discretizing continuous features, and returning labels for the discretized bins.

    This function performs several data preprocessing steps on the given table:
    - Removes specified columns (index 0, 1, 2, 5).
    - Shuffles the rows such that an equal number of 'TRUE' and 'FALSE' values are sampled.
    - Discretizes continuous columns (estimated_diameter_min, estimated_diameter_max, relative_velocity, miss_distance)
      into discrete bins based on their ranges, and replaces the continuous values with the corresponding bin index.

    Args:
        table (MyPyTable): A MyPyTable object containing data to be preprocessed.

    Returns:
        dict: A dictionary containing the labels for each discretized column:
            - "min_labels": Labels for the 'estimated_diameter_min' column bins.
            - "max_labels": Labels for the 'estimated_diameter_max' column bins.
            - "velocity_labels": Labels for the 'relative_velocity' column bins.
            - "miss_labels": Labels for the 'miss_distance' column bins.
    """
    np.random.seed(0)
    space_table = table
    columns_to_remove = [0, 1, 2, 5]
    space_table.column_names = [
        col
        for i, col in enumerate(space_table.column_names)
        if i not in columns_to_remove
    ]
    space_table.data = [
        [value for i, value in enumerate(row) if i not in columns_to_remove]
        for row in space_table.data
    ]

    true_list = []
    false_list = []
    for i, item in enumerate(space_table.data):
        if "True" in item:
            true_list.append(i)
        elif "False" in item:
            false_list.append(i)
    np.random.shuffle(true_list)
    np.random.shuffle(false_list)
    full_sample = []
    true_sample = [space_table.data[i] for i in true_list[:1000]]
    false_sample = [space_table.data[i] for i in false_list[:1000]]
    full_sample.extend(true_sample)
    full_sample.extend(false_sample)
    np.random.shuffle(full_sample)
    space_table.data = full_sample

    # remove outliers from all columns in the data
    for index, column_name in enumerate(space_table.column_names):
        list_of_vals = space_table.get_column(column_name)
        new_column = replace_outliers(list_of_vals)
        # replace all values in the column with the new values (if outliers, replaced with median)
        for pos, row in enumerate(space_table.data):
            row[index] = new_column[pos]

    min_diameter_bins, min_labels = discretization(
        space_table.get_column(space_table.column_names.index("estimated_diameter_min"))
    )

    for instance in space_table.data:
        for i, (lower, upper) in enumerate(min_diameter_bins):
            if (
                lower
                <= instance[space_table.column_names.index("estimated_diameter_min")]
                <= upper
            ):
                instance[space_table.column_names.index("estimated_diameter_min")] = i
                break

    max_diameter_bins, max_labels = discretization(
        space_table.get_column(space_table.column_names.index("estimated_diameter_max"))
    )

    for instance in space_table.data:
        for i, (lower, upper) in enumerate(max_diameter_bins):
            if (
                lower
                <= instance[space_table.column_names.index("estimated_diameter_max")]
                <= upper
            ):
                instance[space_table.column_names.index("estimated_diameter_max")] = i
                break

    velocity_bins, velocity_labels = discretization(
        space_table.get_column(space_table.column_names.index("relative_velocity"))
    )

    for instance in space_table.data:
        for i, (lower, upper) in enumerate(velocity_bins):
            if (
                lower
                <= instance[space_table.column_names.index("relative_velocity")]
                <= upper
            ):
                instance[space_table.column_names.index("relative_velocity")] = i
                break

    miss_bins, miss_labels = discretization(
        space_table.get_column(space_table.column_names.index("miss_distance"))
    )
    for instance in space_table.data:
        for i, (lower, upper) in enumerate(miss_bins):
            if (
                lower
                <= instance[space_table.column_names.index("miss_distance")]
                <= upper
            ):
                instance[space_table.column_names.index("miss_distance")] = i
                break
    return {
        "min_labels": min_labels,
        "max_labels": max_labels,
        "velocity_labels": velocity_labels,
        "miss_labels": miss_labels,
    }


def compute_random_subset(values, num_values):
    """Selects F random attributes from an attribute list

    Args:
        values (list of strs or ints): list of attributes
        num_values (int): how many attributes to keep (F in random forest)

    Returns:
        values_copy (list of strs or ints): random list of attributes that is F long

    Notes:
        - from code done in class in EnsembleFun/main.py
    """
    # let's use np.random.shuffle()
    values_copy = values.copy()
    np.random.shuffle(values_copy)  # inplace
    return values_copy[:num_values]


def stratify_train_test_split(X, y, M):
    # stratified train test split:

    # You will need to write code to produce this, but don't feel like you need to reinvent the wheel.
    # When you hear stratify, think group by! Here are two ideas:

    # 1. If you did the stratified k fold bonus on PA5, you can call this function with k = 3.
    # Use the first train/test run as your split.

    # 2. Add a stratify keyword arg to train_test_split().
    # If it is True, then do a group by. You can split each group and then concatenate the splits

    # M is the number of trees to keep
    myevaluation.stratified_kfold_split(M)
    pass


def concatenate_with_phrase(string_list, join_phrase):
    """
    Concatenates a list of strings with a specified joining phrase.

    Args:
        string_list (list of str): The list of strings to concatenate
        join_phrase (str): The phrase used to join strings (e.g., " AND ", " OR ")

    Returns:
        result (str): A single concatenated string with the joining phrase applied

    Notes:
        The last element is not followed by the joining phrase
    """
    result = ""
    for index, value in enumerate(string_list):
        # need to -1 because max index is length - 1
        if index < len(string_list) - 1:
            result = result + value + join_phrase
        else:
            result = result + value
    return result


def print_dataset_info(pytable):
    # print shape info
    rows, columns = pytable.get_shape()
    print(f"This dataset has {rows} instances and {columns} attributes")

    # collect attribute info
    attr_name = []
    attr_type = []
    data_row = pytable.data[0]
    for index, col in enumerate(pytable.column_names):
        attr_name.append(col)
        attr_type.append(type(data_row[index]))
    col_strings = []
    for index in range(len(attr_name)):
        col_strings.append(attr_name[index] + " is of type " + str(attr_type[index]))
    print(
        "Dataset attribute breakdown: ",
        concatenate_with_phrase(col_strings, " \n "),
        ".",
    )

    # info about class attribute
    class_attr = pytable.column_names[-1]
    possible_class_vals = set(pytable.get_column(class_attr))
    print(
        f"The attribute we are trying to predict is {class_attr}. It can be {len(possible_class_vals)} different classifications: {possible_class_vals}.\n"
    )


def plot_bar_chart(dict_object, data_state):
    plt.figure()
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Is_Hazardous Class ({data_state})")
    for bar in plt.bar(
        dict_object.keys(), dict_object.values(), color="skyblue", edgecolor="black"
    ):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            str(height),
            ha="center",
            va="center",
        )
    plt.show()


def plot_multi_hist(dict_object, data):
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (key, labels) in enumerate(dict_object.items()):
        # Get the column index corresponding to the attribute (min, max, velocity, miss)
        column_index = list(dict_object.keys()).index(
            key
        )  # Get the column index for the attribute

        # Extract the binary data for the current attribute (from the first four columns)
        attribute_data = [
            row[column_index] for row in data
        ]  # Select the relevant column for each row
        label_data = [
            row[-1] == "True" for row in data
        ]  # Get True/False labels (assumed to be in the last column)

        # Get the unique values of the attribute data to determine the number of bins
        num_bins = len(labels)  # Number of bins based on the provided labels

        # Create empty lists to store counts for True, False, and Total counts for each bin
        bin_counts_true = [0] * num_bins
        bin_counts_false = [0] * num_bins
        bin_counts_total = [0] * num_bins

        # Dynamically calculate the bin ranges based on the unique attribute values
        bin_edges = [
            min(attribute_data)
            + i * (max(attribute_data) - min(attribute_data)) / num_bins
            for i in range(num_bins + 1)
        ]

        # Calculate the counts for True, False, and Total labels in each bin
        for i, (attribute_value, label) in enumerate(zip(attribute_data, label_data)):
            for bin_index in range(num_bins):
                if bin_edges[bin_index] <= attribute_value <= bin_edges[bin_index + 1]:
                    if label:
                        bin_counts_true[bin_index] += 1
                    else:
                        bin_counts_false[bin_index] += 1
                    bin_counts_total[
                        bin_index
                    ] += 1  # Increment total for the respective bin
                    break

        # Plot the histogram with separate bars for True, False, and Total counts
        bar_width = 0.25  # Adjusted bar width
        bin_centers = [
            i for i in range(num_bins)
        ]  # Set bin centers dynamically based on number of bins

        # Plot bars for True, False, and Total counts
        axes[idx].bar(
            [x - bar_width for x in bin_centers],
            bin_counts_true,
            width=bar_width,
            color="mediumseagreen",
            label="True",
            edgecolor="black",
            align="center",
        )
        axes[idx].bar(
            bin_centers,
            bin_counts_false,
            width=bar_width,
            color="lightcoral",
            label="False",
            edgecolor="black",
            align="center",
        )
        axes[idx].bar(
            [x + bar_width for x in bin_centers],
            bin_counts_total,
            width=bar_width,
            color="skyblue",
            label="Total",
            edgecolor="black",
            align="center",
        )

        # Add the count annotations on the bars
        for i, (true_count, false_count, total_count) in enumerate(
            zip(bin_counts_true, bin_counts_false, bin_counts_total)
        ):
            axes[idx].text(
                bin_centers[i] - bar_width,
                (true_count + 0.1) / 2,
                str(true_count),
                ha="center",
                va="bottom",
                color="black",
            )
            axes[idx].text(
                bin_centers[i],
                (false_count + 0.1) / 2,
                str(false_count),
                ha="center",
                va="bottom",
                color="black",
            )
            axes[idx].text(
                bin_centers[i] + bar_width,
                (total_count + 0.1) / 2,
                total_count,
                ha="center",
                va="bottom",
                color="black",
            )

        # Set the title and labels
        axes[idx].set_title(f"Histogram of {key} (Processed)")
        axes[idx].set_xticks(bin_centers)  # Dynamically set xticks based on bin_centers
        axes[idx].set_xticklabels(labels)

        # Label the axes
        axes[idx].set_xlabel("Category")
        axes[idx].set_ylabel("Frequency")

        # Add legend
        axes[idx].legend()

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()
