from sklearn import svm
from util import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
n_fold = 5
max_exp = 6

def choose_c(x, y):
    kf = cross_validation.KFold(len(x), n_fold)
#     c_set = [ math.pow(2, i) for i in xrange(-max_exp, max_exp + 1)]
    c_set = np.logspace(start=-max_exp, stop=max_exp, num=2 * max_exp + 1, base=2)
#     c_set = np.logspace(-2, 3.5, 12);
    a_score, p_score, r_score = [[] for i in xrange(len(c_set))], [[] for i in xrange(len(c_set))], [[] for i in xrange(len(c_set))]
    scaler = preprocessing.StandardScaler()
    for train, test in kf:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        scaler.fit(x_train)
        for idx, c in enumerate(c_set):
#             clf = svm.SVC(C=c, kernel='linear')
            clf = svm.LinearSVC(C=c)
            clf.fit(scaler.transform(x_train), y_train)
            y_pred = clf.predict(scaler.transform(x_test))
            acc = accuracy_score(y_test, y_pred)
            a_score[idx].append(acc)
            p_score[idx].append(precision_score(y_test, y_pred))
            r_score[idx].append(recall_score(y_test, y_pred))
            print 'c:{} accuracy: {}'.format(c, acc)
    
    max_val, max_id = -1, -1
    for i in xrange(len(a_score)):
        acc = np.mean(a_score[i])
        if acc > max_val:
            max_val = acc
            max_id = i
        a_score[i] = acc
        p_score[i] = np.mean(p_score[i])
        r_score[i] = np.mean(r_score[i])
    c_star = c_set[max_id]
    print 'best c={}; Acc={}'.format(c_star, max_val)
    plt.clf()
    plt.plot(np.log2(c_set), a_score, marker='o')
    ymin, ymax = plt.ylim()
    plt.plot((np.log2(c_star), np.log2(c_star)), (ymin, max_val), 'k--')
    
    plt.xlabel("log2(C)")
    plt.ylabel("Cross validation score")
#     plt.xscale('log')
#     plt.legend(numpoints=1, loc=4)
    plt.savefig('figs/tuning_c.png')
#     plt.show()
    return c_star

if __name__ == "__main__":
    x, y = load_data(k=2)
    choose_c(x, y)
