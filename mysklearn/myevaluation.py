"""Contains functions for evaluating classifiers
"""

import numpy as np

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
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
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
    if random_state:
        np.random.seed(random_state)

    if shuffle:
        X_new = []
        y_new = []
        unused_indices = list(range(len(X)))
        while unused_indices:
            index = np.random.randint(0, len(unused_indices))
            X_new.append(X[unused_indices[index]])
            y_new.append(y[unused_indices[index]])
            unused_indices.pop(index)
    else:
        X_new = X
        y_new = y

    if test_size < 1:
        test_size = int(np.ceil(len(X) * test_size))

    train_indices = range(0, len(X) - test_size)
    test_indices = range(len(X) - test_size, len(X))

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in test_indices:
        X_test.append(X_new[i])
        y_test.append(y_new[i])
    for i in train_indices:
        X_train.append(X_new[i])
        y_train.append(y_new[i])

    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state:
        np.random.seed(random_state)

    indices = list(range(len(X)))

    if shuffle:
        np.random.shuffle(indices)

    split_lens = []
    items_rem = len(X)
    for i in range(n_splits):
        split_len = int(np.ceil(items_rem / (n_splits - i)))
        split_lens.append(split_len)
        items_rem = items_rem - split_len

    X_test_folds = []
    for i in range(n_splits):
        X_test_folds.append([])
        for _ in range(split_lens[i]):
            X_test_folds[i].append(indices.pop(0))

    X_train_folds = []
    for i, X_test_fold in enumerate(X_test_folds):
        X_train_folds.append([])
        for j in range(len(X)):
            if j not in X_test_fold:
                X_train_folds[i].append(j)

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
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
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state:
        np.random.seed(random_state)

    indices = list(range(len(X)))

    if shuffle:
        np.random.shuffle(indices)

    indices_by_y = {y_val : [] for y_val in set(y)}
    for index in indices:
        indices_by_y[y[index]].append(index)

    reordered_keys = list(indices_by_y.keys()).copy()

    if shuffle:
        np.random.shuffle(reordered_keys)

    reordered_index = []
    for y_val in reordered_keys:
        for index in indices_by_y[y_val]:
            reordered_index.append(index)

    X_test_folds = []
    X_train_folds = []
    for _ in range(n_splits):
        X_test_folds.append([])
        X_train_folds.append(reordered_index.copy())

    for i, index in enumerate(reordered_index):
        X_test_folds[i % len(X_test_folds)].append(index)

    for i, X_train_fold in enumerate(X_train_folds):
        for index in X_test_folds[i]:
            X_train_fold.remove(index)

    return X_train_folds, X_test_folds

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
    """
    if y is None:
        pass

    if n_samples is None:
        n_samples = len(X)

    if random_state:
        np.random.seed(random_state)

    indices = list(range(len(X)))
    indices_out_of_bag = indices.copy()
    indices_sample = []
    for _ in range(n_samples):
        index = indices[np.random.randint(0, len(indices))]
        indices_sample.append(index)
        if index in indices_out_of_bag:
            indices_out_of_bag.remove(index)

    X_sample = []
    y_sample = []
    for index in indices_sample:
        X_sample.append(X[index])
        if y:
            y_sample.append(y[index])

    X_out_of_bag = []
    y_out_of_bag = []
    for index in indices_out_of_bag:
        X_out_of_bag.append(X[index])
        if y:
            y_out_of_bag.append(y[index])

    return X_sample, X_out_of_bag, y_sample if y else None, y_out_of_bag if y else None

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
    matrix = []
    for _ in labels:
        matrix.append([0 for _ in labels])

    for index, y in enumerate(y_pred):
        i = labels.index(y_true[index])
        j = labels.index(y)
        matrix[i][j] += 1

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
    correct = 0
    for i, y in enumerate(y_true):
        if y == y_pred[i]:
            correct += 1
    return correct / len(y_true) if normalize else correct

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
    if labels is None:
        labels = []
        for label in y_true:
            if label not in labels:
                labels.append(label)

    if pos_label is None:
        pos_label = labels[0]

    true_positives = 0
    false_positives = 0
    for i, y in enumerate(y_true):
        if y_pred[i] == pos_label:
            if y == pos_label:
                true_positives += 1
            else:
                false_positives += 1

    if true_positives + false_positives != 0:
        return true_positives / (true_positives + false_positives)
    return 0.0

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
    if labels is None:
        labels = []
        for label in y_true:
            if label not in labels:
                labels.append(label)

    if pos_label is None:
        pos_label = labels[0]

    true_positives = 0
    false_negatives = 0
    for i, y in enumerate(y_true):
        if y == y_pred[i] == pos_label:
            true_positives += 1
        elif y == pos_label and y_pred[i] != pos_label:
            false_negatives += 1

    if true_positives + false_negatives != 0:
        return true_positives / (true_positives + false_negatives)
    return 0.0

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
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if precision + recall != 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0
