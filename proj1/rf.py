from util import load_data
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

n_fold = 5
criteria = ['gini', 'entropy']
performance = 0
relation = 1

def classify(x, y, cv, criterion='gini', n_estimator=10):
    acc, prec, recall = [], [], []
    clf = RandomForestClassifier(criterion=criterion, n_estimators=n_estimator)
    for train, test in cv:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
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
            a, p, r, f = classify(x, y, kf, criterion=criterion, n_estimator=500)
            print 'precision: {}'.format(p)
            print "recall: {}".format(r)
            print "f1: {}".format(f)
            print "accuracy: {}".format(a)
    if relation:
        res = []
        for k in xrange(1, 50 + 1):
            print 'num of trees:{}'.format(k * 10)
            a, p, r, f = classify(x, y, kf, criterion='entropy', n_estimator=k * 10)
            print a, p, r, f
            res.append((a, p, r, f))
        with open('res/rf_trees', 'w') as out:
            for v in res:
                out.write('{},{},{},{}\n'.format(v[0], v[1], v[2], v[3]))
    
if __name__ == "__main__":
    main()
