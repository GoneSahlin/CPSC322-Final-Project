import pickle
import os

from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyDecisionTreeClassifier


def create_classifier():
    fpath = os.path.join("input_data", "joined_data.csv")
    table = MyPyTable().load_from_file(fpath)

    X = table.get_columns([])



if __name__ == "__main__":
    create_classifier()