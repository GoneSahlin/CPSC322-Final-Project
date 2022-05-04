import pickle
import os

from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyDummyClassifier, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyRandomForestClassifier, MySimpleLinearRegressionClassifier


def create_classifier():
    fpath = os.path.join("Data", "joined_data.csv")
    table = MyPyTable().load_from_file(fpath)

    table = table.get_columns(['beer_style', 'beer_abv', 'rating', 'brewery_country', 'brewery_rating'])
    table.remove_rows_with_missing_values()


    X = table.get_columns(['beer_style', 'beer_abv', 'brewery_country', 'brewery_rating']).data
    y = table.get_column('rating')

    dt_clf = MyRandomForestClassifier(M=5, N=10, F=2)
    dt_clf.fit(X, y)

    outfile = open("tree.p", "wb")
    pickle.dump(dt_clf, outfile)
    outfile.close()


if __name__ == "__main__":
    create_classifier()