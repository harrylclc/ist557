from sklearn import lda, qda
from util import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
n_fold = 5
is_lda = 0

x, y = load_data(k=2)
pca = PCA(n_components=10)
pca.fit(x)
# print pca.explained_variance_ratio_
x = pca.transform(x)
# x = pca.fit_transform(x)
# exit()

kf = cross_validation.KFold(x.shape[0], n_fold)
acc, prec, recall = [], [], []
if is_lda:
    clf = lda.LDA()
else:
    clf = qda.QDA()

scaler = preprocessing.StandardScaler()
for train, test in kf:
    print 'iter {}'.format(len(acc))
    x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
    scaler.fit(x_train)
    clf.fit(scaler.transform(x_train), y_train)
    y_pred = clf.predict(scaler.transform(x_test))
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
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
