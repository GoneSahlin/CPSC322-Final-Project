"""Utility functions for mysklearn
"""

import numpy as np
from tabulate import tabulate

from mysklearn.mypytable import MyPyTable

def compute_euclidean_distance(v1, v2):
    """Computes the euclidean distance between v1 and v2
    """
    dists = []
    for i, x in enumerate(v1):
        if isinstance(x, str):
            if x == v2[i]:
                dists.append(0)
            else:
                dists.append(1)
        else:
            dists.append((x - v2[i])**2)
    return np.sqrt(sum(dists))

def select_attribute(instances, attributes):
    weighted_entropies = []
    for attribute in attributes:
        col_index = instances.get_column_index(attribute)
        counts = {}
        p_dict = {}
        for row in instances.data:
            value = row[col_index]
            y_value = row[-1]
            if value not in counts:
                counts[value] = 1
            else:
                counts[value] += 1
            if value not in p_dict:
                p_dict[value] = {}
            if y_value not in p_dict[value]:
                p_dict[value][y_value] = 1
            else:
                p_dict[value][y_value] += 1

        addends = []
        for value, count in counts.items():
            entropy_addends = []
            for y_value, y_count in p_dict[value].items():
                entropy_addends.append(y_count / count * np.log2(y_count / count))
            entropy = -sum(entropy_addends)
            addends.append(count / len(instances.data) * entropy)
        weighted_entropy = sum(addends)
        weighted_entropies.append(weighted_entropy)

    min_index = weighted_entropies.index(min(weighted_entropies))
    return attributes[min_index]

def partition_instances(current_instances, attribute):
    """Partitions the current_instances based on the attibute
    """
    partitions = []
    values = []
    col_index = current_instances.get_column_index(attribute)
    for row in current_instances.data:
        if row[col_index] not in values:
            values.append(row[col_index])
            partitions.append(MyPyTable(current_instances.column_names, [row]))
        else:
            partitions[values.index(row[col_index])].data.append(row)

    return partitions, values

def check_case_1(partition):
    """Checks if all class labels in the partition are the same
    """
    first_elem = partition.data[0][-1]
    for row in partition.data:
        if row[-1] != first_elem:
            return False
    return True

def mode(values):
    """Finds the majority, if tie, then choose alphabetically
    """
    value_counts = {}
    for value in values:
        if value not in value_counts:
            value_counts[value] = 1
        else:
            value_counts[value] += 1
    max_count = 0
    max_keys = []
    for key, value in value_counts.items():
        if value > max_count:
            max_count = value
            max_keys = [key]
        elif value == max_count:
            max_keys.append(key)
    max_keys.sort()
    return max_keys[0]

def tdidt(current_instances, available_attributes, previous_length, possible_values):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes)
    available_attributes.remove(attribute)
    att_node = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions, values = partition_instances(current_instances, attribute)
    # for each partition, repeat unless one of the following occurs (base case)

    # CASE 3?
    if len(partitions) < len(possible_values[attribute].items()):
            column = [current_instances.data[i][-1] for i in range(len(current_instances.data))]
            majority = mode(column)
            att_node = ["Leaf", majority, len(current_instances.data), previous_length]
            return att_node

    for i, partition in enumerate(partitions):
    #    CASE 1: all class labels of the partition are the same => make a leaf node
        if check_case_1(partition):
            val_node = ["Value", values[i], ["Leaf", partition.data[0][-1], len(partition.data), len(current_instances.data)]]
            att_node.append(val_node)
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(available_attributes) == 0:
            column = [partition.data[i][-1] for i in range(len(partition.data))]
            majority = mode(column)
            value_node = ["Value", values[i], ["Leaf", majority, len(partition.data), len(current_instances.data)]]
            att_node.append(value_node)
    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        # elif len(partitions) < len(possible_values[attribute].items()):
        #     print("here")
        #     column = [current_instances.data[i][-1] for i in range(len(current_instances.data))]
        #     majority = mode(column)
        #     att_node = ["Leaf", majority, len(current_instances.data), previous_length]
        else:
            val_node = tdidt(partition, available_attributes.copy(), len(current_instances.data), possible_values)
            att_node.append(["Value", values[i], val_node])
    return att_node

def fit_starter_code(X_train, y_train):
    # TODO: programmatically extract the header (e.g. ["att0",
    # "att1", ...])
    # and extract the attribute domains
    # now, I advise stitching X_train and y_train together
    train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
    header = []
    for i in range(len(train[0])):
        header.append("att" + str(i))
    # next, make a copy of your header... tdidt() is going
    # to modify the list
    available_attributes = header.copy().pop(-1)  # never split on class attribute
    # also: recall that python is pass by object reference
    tree = tdidt(train, available_attributes)
    # note: unit test is going to assert that tree == interview_tree_solution
    # (mind the attribute domain ordering)

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(header, value_list[2], instance)

def tdidt_print(tree, attribute_names, class_name, header, rule):
    info_type = tree[0]
    if info_type == "Leaf":
        rule = rule.removesuffix(" AND ")
        rule += " THEN " + class_name + " = " + tree[1]
        print(rule)
        return
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        new_rule = rule + attribute_names[att_index] + " == " + str(value_list[1]) + " AND "
        tdidt_print(value_list[2], attribute_names, class_name, header, new_rule)

def get_median(values):
    return values[len (values)//2]

def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = np.random.randint(0, n) # Return random integers from low (inclusive) to high (exclusive)
        sample.append(rand_index)
    return sample

def compute_random_subset(values, num_values):
    values_copy = values[:] # shallow copy
    np.random.shuffle(values_copy) # in place
    return values_copy[:num_values]

def select_attribute_forest(instances, attributes, F):
    weighted_entropies = []

    f_attributes = compute_random_subset(attributes, F)

    for attribute in attributes:
        col_index = instances.get_column_index(attribute)
        counts = {}
        p_dict = {}
        for row in instances.data:
            value = row[col_index]
            y_value = row[-1]
            if value not in counts:
                counts[value] = 1
            else:
                counts[value] += 1
            if value not in p_dict:
                p_dict[value] = {}
            if y_value not in p_dict[value]:
                p_dict[value][y_value] = 1
            else:
                p_dict[value][y_value] += 1

        addends = []
        for value, count in counts.items():
            entropy_addends = []
            for y_value, y_count in p_dict[value].items():
                entropy_addends.append(y_count / count * np.log2(y_count / count))
            entropy = -sum(entropy_addends)
            addends.append(count / len(instances.data) * entropy)
        weighted_entropy = sum(addends)
        weighted_entropies.append(weighted_entropy)

    min_index = weighted_entropies.index(min(weighted_entropies))
    return attributes[min_index]

def tdidt_forest(current_instances, available_attributes, previous_length, possible_values, F):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    attribute = select_attribute_forest(current_instances, available_attributes, F)
    available_attributes.remove(attribute)
    att_node = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions, values = partition_instances(current_instances, attribute)
    # for each partition, repeat unless one of the following occurs (base case)

    # CASE 3?
    if len(partitions) < len(possible_values[attribute].items()):
            column = [current_instances.data[i][-1] for i in range(len(current_instances.data))]
            majority = mode(column)
            att_node = ["Leaf", majority, len(current_instances.data), previous_length]
            return att_node

    for i, partition in enumerate(partitions):
    #    CASE 1: all class labels of the partition are the same => make a leaf node
        if check_case_1(partition):
            val_node = ["Value", values[i], ["Leaf", partition.data[0][-1], len(partition.data), len(current_instances.data)]]
            att_node.append(val_node)
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(available_attributes) == 0:
            column = [partition.data[i][-1] for i in range(len(partition.data))]
            majority = mode(column)
            value_node = ["Value", values[i], ["Leaf", majority, len(partition.data), len(current_instances.data)]]
            att_node.append(value_node)
    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        # elif len(partitions) < len(possible_values[attribute].items()):
        #     print("here")
        #     column = [current_instances.data[i][-1] for i in range(len(current_instances.data))]
        #     majority = mode(column)
        #     att_node = ["Leaf", majority, len(current_instances.data), previous_length]
        else:
            val_node = tdidt(partition, available_attributes.copy(), len(current_instances.data), possible_values)
            att_node.append(["Value", values[i], val_node])
    return att_node