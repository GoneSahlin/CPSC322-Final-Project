import numpy as np
from mysklearn import myclassifiers

from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyNaiveBayesClassifier

from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

def high_low_discretizer(y):
    """Discretizes y into high or low
    """
    if y >= 100:
        return "high"
    return "low"

def high_mid_low_discretizer(y):
    """Discretizes y into high, mid, or low
    """
    if y < 50:
        return "low"
    if y < 100:
        return "mid"
    return "high"

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    """Tests MySimpleLinearRegressionClassifier.fit
    """
    x = np.arange(0, 100, .1)
    np.random.seed(0)
    y = 2 * x + np.random.normal(0, 5, len(x))
    X = [[i] for i in x]
    lin_clf = MySimpleLinearRegressionClassifier(high_low_discretizer)
    lin_clf.fit(X, y)
    assert np.isclose(2, lin_clf.regressor.slope, .01)
    assert np.isclose(0, lin_clf.regressor.intercept, atol=.2)

    y2 = -1.5 * x + np.random.normal(0, 5, len(x)) + 50
    X2 = [[i] for i in x]
    lin_clf2 = MySimpleLinearRegressionClassifier(high_mid_low_discretizer)
    lin_clf2.fit(X2, y2)
    assert np.isclose(-1.5, lin_clf2.regressor.slope, .01)
    assert np.isclose(50, lin_clf2.regressor.intercept, atol=.4)

def test_simple_linear_regression_classifier_predict():
    """Tests MySimpleLinearRegressionClassifier.predit
    """
    x = np.arange(0, 100, .1)
    np.random.seed(0)
    y = 4 * x + np.random.normal(0, 4, len(x))
    X = [[i] for i in x]
    lin_clf = MySimpleLinearRegressionClassifier(high_low_discretizer)
    lin_clf.fit(X, y)
    assert ["high"] == lin_clf.predict([[30]])
    assert ["low"] == lin_clf.predict([[20]])

    y2 = -3 * x + np.random.normal(0, 3, len(x)) + 130
    lin_clf2 = MySimpleLinearRegressionClassifier(high_mid_low_discretizer)
    lin_clf2.fit(X, y2)
    assert ["high"] == lin_clf2.predict([[0]])
    assert ["mid"] == lin_clf2.predict([[20]])
    assert ["low"] == lin_clf2.predict([[40]])

def test_kneighbors_classifier_kneighbors():
    """Tests MyKNeighborsClassifier.kneighbors
    """
    np.random.seed(0)

    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test1 = [[2, 1], [0, 0], [3, 1]]

    # train my classifier
    kNN_clf = MyKNeighborsClassifier()
    kNN_clf.fit(X_train_class_example1, y_train_class_example1)
    distances, indices = kNN_clf.kneighbors(X_test1)
    # hand calculations for testing against
    hand_distances = [[1.0, 1.41421356, 1.9465097], [0.0, 0.33, 1.0], [2.0, 2.23606798, 2.85112259]]
    hand_indices = [[0, 1, 2], [3, 2, 1], [0, 1, 2]]
    # compare the two
    assert np.all(np.isclose(distances, hand_distances))
    assert np.all(np.isclose(indices, hand_indices))

    # Example 2
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test2 = [[4, 5], [2, 1], [5, 6], [4, 8]]
    # train my classifier
    kNN_clf2 = MyKNeighborsClassifier()
    kNN_clf2.fit(X_train_class_example2, y_train_class_example2)
    distances2, indices2 = kNN_clf2.kneighbors(X_test2)
    # hand calculations for testing against
    hand_distances2 = [[1.0, 2.23606798, 3.16227766], [1.0, 1.41421356, 1.41421356], [1.0, 2.23606798, 4.0], [2.82842712, 3.60555128, 4.0]]
    hand_indices2 = [[3, 1, 0], [5, 4, 0], [1, 3, 7], [1, 7, 3]]
    # compare the two
    assert np.all(np.isclose(distances2, hand_distances2))
    assert np.all(np.equal(indices2, hand_indices2))

    # Example 3
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    X_test3 = [[10.5, 10.7], [3.1, 19.3], [10.5, 3.2], [8.3, 1.1]]
    # train my classifier
    kNN_clf3 = MyKNeighborsClassifier()
    kNN_clf3.fit(X_train_bramer_example, y_train_bramer_example)
    distances3, indices3 = kNN_clf3.kneighbors(X_test3)
    # hand calculations for testing against
    hand_distances3 = [[1.14017543, 1.52643375, 1.58113883], [5.02493781, 7.65375725, 9.82344135],
                        [3.1144823,  3.80788655, 6.40702739], [4.5, 6.78011799, 8.71435597]]
    hand_indices3 = [[7, 8, 6], [3, 4, 6], [10, 9, 7], [10, 9, 5]]
    # compare the two
    assert np.all(np.isclose(distances3, hand_distances3))
    assert np.all(np.equal(indices3, hand_indices3))

def test_kneighbors_classifier_predict():
    """Tests MyKNeighborsClassifier.predict
    """
    np.random.seed(0)

    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test1 = [[2, 1], [0, 0], [3, 1]]

    # train my classifier
    kNN_clf = MyKNeighborsClassifier()
    kNN_clf.fit(X_train_class_example1, y_train_class_example1)
    y_predicted = kNN_clf.predict(X_test1)
    # hand calculations for testing against
    hand_predicted = ['bad', 'good', 'bad']
    # compare the two
    assert np.all([y_predicted[i] == hand_predicted[i] for i in range(len(y_predicted))])

    # Example 2
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test2 = [[4, 5], [2, 1], [5, 6], [4, 8]]
    # train my classifier
    kNN_clf2 = MyKNeighborsClassifier()
    kNN_clf2.fit(X_train_class_example2, y_train_class_example2)
    y_predicted2 = kNN_clf2.predict(X_test2)
    # hand calculations for testing against
    hand_predicted2 = ['no', 'no', 'yes', 'yes']
    # compare the two
    assert np.all([y_predicted2[i] == hand_predicted2[i] for i in range(len(y_predicted2))])

    # Example 3
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    X_test3 = [[10.5, 10.7], [3.1, 19.3], [10.5, 3.2], [8.3, 1.1]]
    # train my classifier
    kNN_clf3 = MyKNeighborsClassifier()
    kNN_clf3.fit(X_train_bramer_example, y_train_bramer_example)
    y_predicted3 = kNN_clf3.predict(X_test3)
    # hand calculations for testing against
    hand_predicted3 = ['+', '-', '+', '+']
    # compare the two
    assert np.all([y_predicted3[i] == hand_predicted3[i] for i in range(len(y_predicted3))])

def test_dummy_classifier_fit():
    """Tests MyDummyClassifier.fit
    """
    X_train = [[]]  # not used by dummy classifier
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    assert dummy_clf.most_common_label == "yes"

    # Test 2
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_clf.fit(X_train, y_train)
    assert dummy_clf.most_common_label == "no"

    # Test 3
    y_train = list(np.random.choice(["1", "2", "3", "4"], 100, replace=True, p=[0.2, 0.2, 0.2, 0.4]))
    dummy_clf.fit(X_train, y_train)
    assert dummy_clf.most_common_label == '4'

def test_dummy_classifier_predict():
    """Tests MyDummyClassifier.predict
    """
    X_train = [[]]  # not used by dummy classifier
    X_test = [[2, 3]]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    y_predicted = dummy_clf.predict(X_test)
    assert y_predicted == ["yes"]

    # Test 2
    X_test = [[1], [2], [3]]
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_clf.fit(X_train, y_train)
    y_predicted = dummy_clf.predict(X_test)
    assert ["no", "no", "no"] == y_predicted
    # Test 3
    X_test = [[1], [2], [3], [4], [5]]
    y_train = list(np.random.choice(["1", "2", "3", "4"], 100, replace=True, p=[0.2, 0.2, 0.2, 0.4]))
    dummy_clf.fit(X_train, y_train)
    y_predicted = dummy_clf.predict(X_test)
    assert ["4", "4", "4", "4", "4"] == y_predicted


def test_naive_bayes_classifier_fit():
    """Tests MyNaiveBayesClassifier.fit
    """
    # in-class Naive Bayes example (lab task #1)
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    hand_priors_1 = {"yes": 0.625, "no": 0.375}
    hand_posteriors_1 = {0: {1: {"yes": .8, "no": 2/3}, 2: {"yes": .2, "no": 1/3}}, 1: {5: {"yes": .4, "no":2/3}, 6: {"yes": 3/5, "no": 1/3}}}

    nb_clf_1 = MyNaiveBayesClassifier()
    nb_clf_1.fit(X_train_inclass_example, y_train_inclass_example)

    for key, value in hand_priors_1.items():
        assert np.allclose(value, nb_clf_1.priors[key])

    for key_1, value_1 in hand_posteriors_1.items():
        for key_2, value_2 in value_1.items():
            for key_3, value_3 in value_2.items():
                assert np.allclose(value_3, nb_clf_1.posteriors[key_1][key_2][key_3])


    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    hand_priors_2 = {"yes": 10/15, "no": 5/15}
    hand_posteriors_2 = {0: {1: {"yes": .2, "no": .6}, 2: {"yes": .8, "no": .4}}, 1: {1: {"yes": .3, "no":.2}, 2: {"yes": .4, "no": .4},
                        3: {"yes": .3, "no": .4}}, 2: {"fair": {"yes": .7, "no": .4}, "excellent": {"yes": .3, "no": .6}}}

    iphone_pytable = MyPyTable(iphone_col_names, iphone_table)
    X_train_iphone = iphone_pytable.get_columns(["standing", "job_status", "credit_rating"])
    y_train_iphone = iphone_pytable.get_column("buys_iphone")

    nb_clf_2 = MyNaiveBayesClassifier()
    nb_clf_2.fit(X_train_iphone.data, y_train_iphone)

    for key, value in hand_priors_2.items():
        assert np.allclose(value, nb_clf_2.priors[key])

    for key_1, value_1 in hand_posteriors_2.items():
        for key_2, value_2 in value_1.items():
            for key_3, value_3 in value_2.items():
                assert np.allclose(value_3, nb_clf_2.posteriors[key_1][key_2][key_3])

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]

    hand_priors_3 = {"on time": 14/20, "late": .1, "very late": 3/20, "cancelled": .05}
    hand_posteriors_3 = {0: {"weekday": {"on time": 9/14, "late": .5, "very late": 1, "cancelled": 0},
                        "saturday": {"on time": 2/14, "late": .5, "very late": 0, "cancelled": 1},
                         "sunday": {"on time": 1/14, "late": 0, "very late": 0, "cancelled": 0},
                         "holiday": {"on time": 2/14, "late": 0, "very late": 0, "cancelled": 0}},
                         1: {"spring": {"on time": 4/14, "late": 0, "very late": 0, "cancelled": 1},
                         "summer": {"on time": 6/14, "late": 0, "very late": 0, "cancelled": 0},
                         "autumn": {"on time": 2/14, "late": 0, "very late": 1/3, "cancelled": 0},
                         "winter": {"on time": 2/14, "late": 1, "very late": 2/3, "cancelled": 0}},
                         2: {"none": {"on time": 5/14, "late": 0, "very late": 0, "cancelled": 0},
                         "normal": {"on time": 5/14, "late": .5, "very late": 2/3, "cancelled": 0},
                         "high": {"on time": 4/14, "late": .5, "very late": 1/3, "cancelled": 1}},
                         3: {"none": {"on time": 5/14, "late": .5, "very late": 1/3, "cancelled": 0},
                         "slight": {"on time": 8/14, "late": 0, "very late": 0, "cancelled": 0},
                         "heavy": {"on time": 1/14, "late": .5, "very late": 2/3, "cancelled": 1}}}

    bramer_pytable = MyPyTable(train_col_names, train_table)
    X_train_bramer = bramer_pytable.get_columns(["day", "season", "wind", "rain"])
    y_train_bramer = bramer_pytable.get_column("class")

    nb_clf_3 = MyNaiveBayesClassifier()
    nb_clf_3.fit(X_train_bramer.data, y_train_bramer)

    for key, value in hand_priors_3.items():
        assert np.allclose(value, nb_clf_3.priors[key])

    for key_1, value_1 in hand_posteriors_3.items():
        for key_2, value_2 in value_1.items():
            for key_3, value_3 in value_2.items():
                assert np.allclose(value_3, nb_clf_3.posteriors[key_1][key_2][key_3])

def test_naive_bayes_classifier_predict():
    """Tests MyNaiveBayesClassifier.predict
    """
    # in-class Naive Bayes example (lab task #1)
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    X_test_inclass_example = [[1, 5]]
    y_test_inclass_example = ["yes"]

    nb_clf_1 = MyNaiveBayesClassifier()
    nb_clf_1.fit(X_train_inclass_example, y_train_inclass_example)

    y_pred_1 = nb_clf_1.predict(X_test_inclass_example)

    assert np.all([y_test_inclass_example[i] == y_pred_1[i] for i in range(len(y_test_inclass_example))])

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    X_test_iphone = [[2,2,"fair"],[1,1,"excellent"]]
    y_test_iphone = ["yes", "no"]

    iphone_pytable = MyPyTable(iphone_col_names, iphone_table)
    X_train_iphone = iphone_pytable.get_columns(["standing", "job_status", "credit_rating"])
    y_train_iphone = iphone_pytable.get_column("buys_iphone")

    nb_clf_2 = MyNaiveBayesClassifier()
    nb_clf_2.fit(X_train_iphone.data, y_train_iphone)
    y_pred_2 = nb_clf_2.predict(X_test_iphone)

    assert np.all([y_test_iphone[i] == y_pred_2[i] for i in range(len(y_test_iphone))])

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]

    X_test_bramer = [["weekday", "winter", "high", "heavy"]]
    y_test_bramer = ["very late"]

    bramer_pytable = MyPyTable(train_col_names, train_table)
    X_train_bramer = bramer_pytable.get_columns(["day", "season", "wind", "rain"])
    y_train_bramer = bramer_pytable.get_column("class")

    nb_clf_3 = MyNaiveBayesClassifier()
    nb_clf_3.fit(X_train_bramer.data, y_train_bramer)
    y_pred_3 = nb_clf_3.predict(X_test_bramer)

    assert np.all([y_test_bramer[i] == y_pred_3[i] for i in range(len(y_test_bramer))])

def test_decision_tree_classifier_fit():
    header_interview = ["level", "lang", "tweets", "phd"]

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

    tree_interview =   ['Attribute', 'att0',
                                    ['Value', 'Senior',
                                        ['Attribute', 'att2',
                                            ['Value', 'no',
                                                ['Leaf', 'False', 3, 5]
                                            ],
                                            ['Value', 'yes',
                                                ['Leaf', 'True', 2, 5]
                                            ]
                                        ]
                                    ],
                                    ['Value', 'Mid',
                                        ['Leaf', 'True', 4, 14]
                                    ],
                                    ['Value', 'Junior',
                                        ['Attribute', 'att3',
                                            ['Value', 'no',
                                                ['Leaf', 'True', 3, 5]
                                            ],
                                            ['Value', 'yes',
                                                ['Leaf', 'False', 2, 5]
                                            ]
                                        ]
                                    ]
                                ]
    dt_interview = MyDecisionTreeClassifier()
    dt_interview.fit(X_train_interview, y_train_interview)
    assert str(tree_interview) == str(dt_interview.tree)

    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ['A', 'B', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'A', 'B', 'B', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'A', 'A', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'B', 'B'],
        ['B', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'B', 'A', 'A'],
        ['B', 'B', 'B', 'A', 'A'],
        ['B', 'B', 'A', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['B', 'A', 'B', 'A', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'B', 'A', 'B', 'B'],
        ['B', 'A', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B']
    ]
    y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                      'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                      'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                      'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                      'SECOND', 'SECOND']

    tree_degrees =  ['Attribute', 'att0',
                        ['Value', 'A',
                            ['Attribute', 'att4',
                                ['Value', 'B',
                                    ['Attribute', 'att3',
                                        ['Value', 'B',
                                            ['Leaf', 'SECOND', 7, 9]
                                        ],
                                        ['Value', 'A',
                                            ['Attribute', 'att1',
                                                ['Value', 'B',
                                                    ['Leaf', 'SECOND', 1, 2]
                                                ],
                                                ['Value', 'A',
                                                    ['Leaf', 'FIRST', 1, 2]
                                                ]
                                            ]
                                        ]
                                    ]
                                ],
                                ['Value', 'A',
                                    ['Leaf', 'FIRST', 5, 14]
                                ]
                            ]
                        ],
                        ['Value', 'B',
                            ['Leaf', 'SECOND', 12, 26]
                        ]
                    ]

    dt_degrees = MyDecisionTreeClassifier()
    dt_degrees.fit(X_train_degrees, y_train_degrees)
    assert str(tree_degrees) == str(dt_degrees.tree)

    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    tree_iphone =   ['Attribute', 'att0',
                        ['Value', 1,
                            ['Attribute', 'att1',
                                ['Value', 3,
                                    ['Leaf', 'no', 2, 5]
                                ],
                                ['Value', 2,
                                    ['Attribute', 'att2',
                                        ['Value', 'fair',
                                            ['Leaf', 'no', 1, 2]
                                        ],
                                        ['Value', 'excellent',
                                            ['Leaf', 'yes', 1, 2]
                                        ]
                                    ]
                                ],
                                ['Value', 1,
                                    ['Leaf', 'yes', 1, 5]
                                ]
                            ]
                        ],
                        ['Value', 2,
                            ['Attribute', 'att2',
                                ['Value', 'fair',
                                    ['Leaf', 'yes', 6, 10]
                                ],
                                ['Value', 'excellent',
                                    ['Attribute', 'att1',
                                        ['Value', 1,
                                            ['Leaf', 'no', 2, 4]
                                        ],
                                        ['Value', 2,
                                            ['Leaf', 'no', 2, 4]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]

    dt_iphone = MyDecisionTreeClassifier()
    dt_iphone.fit(X_train_iphone, y_train_iphone)
    # assert str(dt_iphone.tree) == str(tree_iphone)

def test_decision_tree_classifier_predict():
    # interview dataset
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

    X_test_interview = [["Junior", "Java", "yes", "no"],["Junior", "Java", "yes", "yes"]]
    y_desk_interview = ["True", "False"]

    dt_interview = MyDecisionTreeClassifier()
    dt_interview.fit(X_train_interview, y_train_interview)
    y_predicted = dt_interview.predict(X_test_interview)
    assert str(y_predicted) == str(y_desk_interview)

    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ['A', 'B', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'A', 'B', 'B', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'A', 'A', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'B', 'B'],
        ['B', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'B', 'A', 'A'],
        ['B', 'B', 'B', 'A', 'A'],
        ['B', 'B', 'A', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['B', 'A', 'B', 'A', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'B', 'A', 'B', 'B'],
        ['B', 'A', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B']
    ]
    y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                      'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                      'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                      'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                      'SECOND', 'SECOND']

    X_test_degrees = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    y_desk_degrees = ["SECOND", "FIRST", "FIRST"]

    dt_degrees = MyDecisionTreeClassifier()
    dt_degrees.fit(X_train_degrees, y_train_degrees)
    y_predicted_degrees = dt_degrees.predict(X_test_degrees)
    assert str(y_predicted_degrees) == str(y_desk_degrees)

        # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    X_test_iphone = [[2,2,"fair"],[1,1,"excellent"]]
    y_desk_iphone = ["yes", "yes"]

    dt_iphone = MyDecisionTreeClassifier()
    dt_iphone.fit(X_train_iphone, y_train_iphone)
    y_predicted_iphone = dt_iphone.predict(X_test_iphone)
    assert str(y_predicted_iphone) == str(y_desk_iphone)

def test_random_forest_classifier_fit():
    np.random.seed(0)
    # interview dataset
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

    X_test_interview = [["Junior", "Java", "yes", "no"],["Junior", "Java", "yes", "yes"]]
    y_desk_interview = ["True", "False"]

    rf_clf = MyRandomForestClassifier(7, 20, 2)
    rf_clf.fit(X_train_interview, y_train_interview)

    random_forest_actual = [['Attribute', 'att0', ['Value', 'Mid', ['Leaf', 'True', 4, 14]], ['Value', 'Junior', ['Leaf', 'True', 7, 14]], ['Value', 'Senior', ['Leaf', 'False', 3, 14]]], ['Attribute', 'att2', ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Mid', ['Leaf', 'True', 2, 9]], ['Value', 'Senior', ['Leaf', 'True', 3, 9]], ['Value', 'Junior', ['Leaf', 'True', 4, 9]]]], ['Value', 'no', ['Leaf', 'False', 5, 14]]], ['Attribute', 'att1', ['Value', 'R', ['Leaf', 'False', 2, 14]], ['Value', 'Java', ['Leaf', 'False', 4, 14]], ['Value', 'Python', ['Attribute', 'att0', ['Value', 'Mid', ['Leaf', 'True', 1, 8]], ['Value', 'Junior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 5, 6]], ['Value', 'yes', ['Leaf', 'True', 1, 6]]]], ['Value', 'Senior', ['Leaf', 'False', 1, 8]]]]], ['Attribute', 'att0', ['Value', 'Senior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'False', 5, 6]], ['Value', 'yes', ['Leaf', 'True', 1, 6]]]], ['Value', 'Junior', ['Leaf', 'True', 4, 14]], ['Value', 'Mid', ['Leaf', 'True', 4, 14]]], ['Attribute', 'att2', ['Value', 'yes', ['Leaf', 'True', 9, 14]], ['Value', 'no', ['Leaf', 'False', 5, 14]]], ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'True', 7, 14]], ['Value', 'Java', ['Leaf', 'False', 2, 14]],
['Value', 'R', ['Leaf', 'True', 5, 14]]], ['Attribute', 'att2', ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Mid', ['Leaf', 'True', 1, 9]], ['Value', 'Junior', ['Leaf', 'False', 6, 9]], ['Value', 'Senior', ['Leaf', 'True', 2, 9]]]], ['Value', 'no', ['Leaf', 'False', 5, 14]]]]

    assert random_forest_actual == rf_clf.trees
    assert len(rf_clf.trees) == 7

def test_random_forest_classifier_predict():
    np.random.seed(0)
    # interview dataset
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

    X_test_interview = [["Junior", "Java", "yes", "no"],["Junior", "Java", "yes", "yes"]]
    y_desk_interview = ["True", "True"]

    rf_clf = MyRandomForestClassifier(7, 20, 2)
    rf_clf.fit(X_train_interview, y_train_interview)
    y_predicted_interview = rf_clf.predict(X_test_interview)
    assert y_desk_interview == y_predicted_interview
