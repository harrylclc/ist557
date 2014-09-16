from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from util import load_data

n_fold = 5

criteria = ['gini', 'entropy']
performance = 0
relation = 1

def classify(x, y, cv, n_estimator=50):
    acc, prec, recall = [], [], []
    base_clf = DecisionTreeClassifier()
    clf = AdaBoostClassifier(n_estimators=n_estimator)
    for train, test in cv:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
    a = np.mean(acc)
    p = np.mean(prec)
    r = np.mean(recall)
    f = 2 * p * r / (p + r)
    return a, p, r, f

def main():
    x, y = load_data(k=2)
    kf = cross_validation.KFold(len(x), n_fold)
    a, p, r, f = classify(x, y, kf, n_estimator=10)
    print 'precision: {}'.format(p)
    print "recall: {}".format(r)
    print "f1: {}".format(f)
    print "accuracy: {}".format(a)
    
if __name__ == "__main__":
    main()
