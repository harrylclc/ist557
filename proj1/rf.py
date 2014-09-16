from util import load_data
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

n_fold = 5
criteria = ['gini', 'entropy']
performance = 0

def classify(x, y, cv, criterion='gini', n_esimator=10):
    acc, prec, recall = [], [], []
    for train, test in cv:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        clf = RandomForestClassifier(criterion=criterion, n_estimators=500)
        clf.fit(x_train, y_train)
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
    if performance:
        for criterion in criteria:
            print 'criterion: {}'.format(criterion)
            a, p, r, f = classify(x, y, kf, criterion=criterion)
            print 'precision: {}'.format(p)
            print "recall: {}".format(r)
            print "f1: {}".format(f)
            print "accuracy: {}".format(a)
    
if __name__ == "__main__":
    main()
