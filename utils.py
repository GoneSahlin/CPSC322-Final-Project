from mysklearn.mypytable import MyPyTable
import mysklearn.myevaluation as myevaluation
from tabulate import tabulate


def high_low_discretizer(val):
    if val >= 3.5:
        return "high"
    return "low"

def generalize_style(style_in):
    pass

def print_confusion_matrix(confusion_matrix, labels):
    """Prints out the confusion matrix nicely
    """
    headers = ['']
    headers.extend(labels)
    headers.extend(['Total', 'Recognition (%)'])

    # create index
    confusion_table = MyPyTable(column_names=labels, data=confusion_matrix)
    confusion_table.add_column('', labels, 0)

    # calculate total and recognition
    totals = []
    recognitions = []
    for i, row in enumerate(confusion_table.data):
        total = sum(row[1:])
        totals.append(total)
        recognitions.append(round(row[i+1] / total * 100) if total != 0 else 0)

    confusion_table.add_column('Total', totals)
    confusion_table.add_column('Recognition %', recognitions)

    print(tabulate(confusion_table.data, headers=confusion_table.column_names))


def measure_performance(y_true, y_pred, labels, pos_label, name):
    acc = myevaluation.accuracy_score(y_true, y_pred)
    err = 1 - acc
    prec = myevaluation.binary_precision_score(y_true, y_pred, labels, pos_label)
    rec = myevaluation.binary_recall_score(y_true, y_pred, labels, pos_label)
    f1 = myevaluation.binary_f1_score(y_true, y_pred, labels, pos_label)
    cmat = myevaluation.confusion_matrix(y_true, y_pred, labels)

    print(name + " Perfomance:")
    print("Accuracy:", acc)
    print("Error rate:", err)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 measure:", f1)
    print("Confusion Matrix:")
    print_confusion_matrix(cmat, labels)