"""
Programmer: Michael D'Arcy-Evans
Class: CPSC322-01, Fall 2024
Programming Assignment #6
11/5/2024
I attempted the bonus.

Description: Reused utility functions
"""

import numpy as np
from graphviz import Graph


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
            if available_attributes:
                return tdidt(
                    current_instances,
                    available_attributes,
                    header,
                    attribute_domain,
                )
            else:
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
