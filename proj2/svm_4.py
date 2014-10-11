from sklearn import svm
from util import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
n_fold = 5

x, y = load_data(k=3)
kf = cross_validation.KFold(x.shape[0], n_fold)
acc, prec, recall = [], [], []

scaler = preprocessing.StandardScaler()

for train, test in kf:
    print 'iter {}'.format(len(acc))
    x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
    scaler.fit(x_train)
    clf = svm.SVC(gamma=1/72.)
    clf.fit(scaler.transform(x_train), y_train)
    y_pred = clf.predict(scaler.transform(x_test))
    acc.append(accuracy_score(y_test, y_pred))
    prec.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    print acc
a = np.mean(acc)
p = np.mean(prec)
r = np.mean(recall)
f = 2 * p * r / (p + r)

print 'precision: {}'.format(p)
print "recall: {}".format(r)
print "f1: {}".format(f)
print "accuracy: {}".format(a)
