"""Contains my different classifiers
"""

import numpy as np

from mysklearn.mypytable import MyPyTable
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

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
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()

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
        y_predicted = [self.discretizer(y) for y in self.regressor.predict(X_test)]

        return y_predicted

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
        for instance in X_test:
            test_distances_indices = [(myutils.compute_euclidean_distance(instance, x), i) for i, x in enumerate(self.X_train)]
            test_distances_indices.sort()
            # select n closest neighbors
            test_distances = [x[0] for x in test_distances_indices]
            test_indices = [x[1] for x in test_distances_indices]

            instance_distances = []
            instance_neighbor_indices = []
            for _ in range(self.n_neighbors):
                min_dist = test_distances[0]
                i = 0
                while test_distances[i] == min_dist:
                    i += 1
                index = np.random.randint(0, i)
                instance_distances.append(test_distances.pop(index))
                instance_neighbor_indices.append(test_indices.pop(index))

            distances.append(instance_distances)
            neighbor_indices.append(instance_neighbor_indices)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, indices = self.kneighbors(X_test)
        y_predicted = []
        for i in range(len(X_test)):
            y = [self.y_train[index] for index in indices[i]]
            counts_dict = {}
            for value in y:
                if value not in counts_dict:
                    counts_dict[value] = 1
                else:
                    counts_dict[value] += 1
            # find max value
            max_count = 0
            max_key = []
            for key, value in counts_dict.items():
                if value > max_count:
                    max_count = value
                    max_key = key
            y_predicted.append(max_key)

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
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

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
        count_dict = {}
        for y in y_train:
            if y not in count_dict:
                count_dict[y] = 1
            else:
                count_dict[y] += 1
        max_count = 0
        max_keys = []
        for key, value in count_dict.items():
            if value == max_count:
                max_keys.append(key)
            elif value > max_count:
                max_count = value
                max_keys = [key]
        self.most_common_label = max_keys[np.random.randint(0, len(max_keys))]  # random if tied

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [self.most_common_label for _ in range(len(X_test))]
        return y_predicted


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict, key is y_label): The prior probabilities computed for each
            label in the training set.
        posteriors(3x nested dict, 1st key is att, 2nd key is x_label, 3rd key is y_label): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
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
        # priors
        y_labels = []
        for y in y_train:
            if y not in y_labels:
                y_labels.append(y)

        self.priors = {}
        for y_label in y_labels:
            self.priors[y_label] = sum([1 if y == y_label else 0 for y in y_train]) / len(y_train)

        #posteriors
        X_train_table = MyPyTable(list(range(len(X_train[0]))), X_train)
        self.posteriors = {}
        for att in X_train_table.column_names:
            col = X_train_table.get_column(att)
            # col = [X_train_table.data[i][att] for i in range(len(X_train_table.data))]
            col_labels = []
            for x in col:
                if x not in col_labels:
                    col_labels.append(x)

            self.posteriors[att] = {}

            for x_label in col_labels:
                self.posteriors[att][x_label] = {}

                for y_label in y_labels:
                    correct_y_indexes = []
                    for i, y in enumerate(y_train):
                        if y_label == y:
                            correct_y_indexes.append(i)

                    count = sum([1 if x_label == col[index] else 0 for index in correct_y_indexes])

                    self.posteriors[att][x_label][y_label] = count / len(correct_y_indexes)



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in X_test:
            y_labels = []
            probs = []
            for y_label in self.priors:
                y_labels.append(y_label)

                prob = 1
                for i, x_label in enumerate(x):
                    if x_label in self.posteriors[i]:
                        prob *= self.posteriors[i][x_label][y_label]

                prob *= self.priors[y_label]
                probs.append(prob)

            max_prob = max(probs)
            max_indices = []
            for i, prob in enumerate(probs):
                if prob == max_prob:
                    max_indices.append(i)

            if len(max_indices) == 1:
                y_predicted.append(y_labels[max_indices[0]])
            else:
                index = np.random.choice(max_indices)
                y_predicted.append(y_labels[index])

        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

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

        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.header = []
        for i in range(len(train[0])):
            self.header.append("att" + str(i))
        train_table = MyPyTable(self.header, train)

        # find all possible values for each attribute
        possible_values = {}
        for attribute in self.header:
            possible_values[attribute] = {}
        for row in X_train:
            for index, value in enumerate(row):
                if value not in possible_values[self.header[index]]:
                    possible_values[self.header[index]][value] = True

        # next, make a copy of your header... tdidt() is going
        # to modify the list
        available_attributes = self.header.copy()
        available_attributes.pop(-1)  # never split on class attribute

        # also: recall that python is pass by object reference
        self.tree = myutils.tdidt(train_table, available_attributes, len(train_table.data), possible_values)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            y_predicted.append(myutils.tdidt_predict(self.header, self.tree, instance))
        return y_predicted

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
        if attribute_names is None:
            attribute_names = self.header[:-1]

        myutils.tdidt_print(self.tree, attribute_names, class_name, self.header, "IF ")

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
        pass # TODO: (BONUS) fix this


class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        M(int): The number of trees selected.
        N(int): The total number of trees generated.
        F(int): The number of randomly selected attributes to randomly split on.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, M, N, F):
        """Initializer for MyRandomForestClassifier.

        Args:
            M(int): The number of trees selected.
            N(int): The total number of trees generated.
            F(int): The number of randomly selected attributes to randomly split on.
        """
        self.X_train = None
        self.y_train = None
        self.M = M
        self.N = N
        self.F = F

    def fit(self, X_train, y_train):
        """Fits a random forest classifier to X_train and y_train

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = y_train
        self.trees = []

        self.header = []
        for i in range(len(X_train[0])):
            self.header.append("att" + str(i))

        # find all possible values for each attribute
        possible_values = {}
        for attribute in self.header:
            possible_values[attribute] = {}
        for row in X_train:
            for index, value in enumerate(row):
                if value not in possible_values[self.header[index]]:
                    possible_values[self.header[index]][value] = True

        tree_accuracies = []
        # for each decision tree
        for i in range(self.N):
            sample = myutils.compute_bootstrapped_sample(self.X_train)
            train = [self.X_train[index] +  [self.y_train[index]] for index in sample]

            train_table = MyPyTable(self.header, train)
            available_attributes = self.header.copy()
            available_attributes.pop(-1)  # never split on class attribute

            tree = myutils.tdidt_forest(train_table, available_attributes, len(train_table.data), possible_values, self.F)
            self.trees.append(tree)

            # test tree
            validation = list(range(len(self.X_train)))
            for index in sample:
                if index in validation:
                    validation.remove(index)

            correct = 0  # number of validation instances predicted correctly
            for index in validation:
                X_instance = self.X_train[index]
                y_instance = self.y_train[index]
                y_pred = myutils.tdidt_predict(self.header, tree, X_instance)
                if y_pred == y_instance:
                    correct += 1

            accuracy = correct / len(validation)
            tree_accuracies.append(accuracy)

        # find index of m highest accuracies
        m_indexes = sorted(range(len(tree_accuracies)), key=lambda i: tree_accuracies[i], reverse=True)[:self.M]

        # remove trees not selected
        for i in range(len(self.trees) - 1, -1, -1):  # iterate backwards, so pop() doesn't affect next index
            if i not in m_indexes:
                self.trees.pop()


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            tree_predictions = {}
            for tree in self.trees:
                prediction = myutils.tdidt_predict(self.header, tree, instance)
                if prediction in tree_predictions:
                    tree_predictions[prediction] += 1
                else:
                    tree_predictions[prediction] = 1
            majority_prediction = max(tree_predictions, key=tree_predictions.get)
            if majority_prediction is None:
                majority_prediction = prediction.keys()[0]
            y_predicted.append(majority_prediction)
        return y_predicted


