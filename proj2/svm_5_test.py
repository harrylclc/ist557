from sklearn import svm
from util import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
import numpy as np
from sklearn import preprocessing
import math
n_fold = 5
max_exp = 4

def choose_c_gamma(x, y):
#     kf = cross_validation.KFold(len(x), n_fold)
    kf = cross_validation.StratifiedKFold(y, n_fold)
#     c_set = np.logspace(start=-max_exp, stop=max_exp, num=2 * max_exp + 1, base=2)
    c_set = [ math.pow(2, i) for i in xrange(-4, 5)]
    gamma_set = [ math.pow(2, i) for i in xrange(-8, 5)]
#     gamma_set = np.logspace(start=-max_exp, stop=max_exp, num=2 * max_exp + 1, base=2)
    a_score = [[[] for j in xrange(len(gamma_set))] for i in xrange(len(c_set))]
    scaler = preprocessing.StandardScaler()
    k_in = 0
    for train, test in kf:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        scaler.fit(x_train)
        print "Out:{} In:{}".format(k_out, k_in)
        for c_idx, c in enumerate(c_set):
            for g_idx , gamma in enumerate(gamma_set):
                clf = svm.SVC(C=c, gamma=gamma)
                clf.fit(scaler.transform(x_train), y_train)
                y_pred = clf.predict(scaler.transform(x_test))
                acc = accuracy_score(y_test, y_pred)
                a_score[c_idx][g_idx].append(acc)
                print 'c:{} g:{} accuracy: {}'.format(c, gamma, acc)
        k_in += 1
    max_val, max_cid, max_gid = -1, -1, -1
    for i in xrange(len(c_set)):
        for j in xrange(len(gamma_set)):
            acc = np.mean(a_score[i][j])
            if acc > max_val:
                max_val = acc
                max_cid = i
                max_gid = j
            a_score[i][j] = acc
    c_star = c_set[max_cid]
    gamma_star = gamma_set[max_gid]
    return c_star, gamma_star

def main():
    global k_out
    k_out = 0
    x, y = load_data(k=3)
    print 'sampled example: {}'.format(len(y))
    print 'positive examples: {}'.format(np.count_nonzero(y))
#     exit()
    kf = cross_validation.KFold(len(x), n_fold)
    scaler = preprocessing.StandardScaler()
    acc, prec, recall = [], [], []
    for train, test in kf:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        c_star, gamma_star = choose_c_gamma(x_train, y_train)
        print '=========c*:{} g*:{}'.format(c_star, gamma_star)
        scaler.fit(x_train)
        clf = svm.SVC(C=c_star, gamma=gamma_star)
        clf.fit(scaler.transform(x_train), y_train)
        y_pred = clf.predict(scaler.transform(x_test))
        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        print acc
        k_out += 1
    a = np.mean(acc)
    p = np.mean(prec)
    r = np.mean(recall)
    f = 2 * p * r / (p + r)
    
    print 'precision: {}'.format(p)
    print "recall: {}".format(r)
    print "f1: {}".format(f)
    print "accuracy: {}".format(a)
      
if __name__ == "__main__":
    main()
