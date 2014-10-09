from sklearn import svm
from util import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
import numpy as np
from sklearn import preprocessing
n_fold = 5
max_exp = 4

def choose_c(x, y):
    kf = cross_validation.KFold(len(x), n_fold)
    c_set = np.logspace(start=-max_exp, stop=max_exp, num=2 * max_exp + 1, base=2)
    a_score, p_score, r_score = [[] for i in xrange(len(c_set))], [[] for i in xrange(len(c_set))], [[] for i in xrange(len(c_set))]
    scaler = preprocessing.StandardScaler()
    k_in = 0
    for train, test in kf:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        scaler.fit(x_train)
        print "Out:{} In:{}".format(k_out, k_in)
        for idx, c in enumerate(c_set):
            clf = svm.LinearSVC(C=c)
            clf.fit(scaler.transform(x_train), y_train)
            y_pred = clf.predict(scaler.transform(x_test))
            acc = accuracy_score(y_test, y_pred)
            a_score[idx].append(acc)
            p_score[idx].append(precision_score(y_test, y_pred))
            r_score[idx].append(recall_score(y_test, y_pred))
            print 'c:{} accuracy: {}'.format(c, acc)
        k_in += 1
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
    return c_star

def main():
    global k_out
    k_out = 0
    x, y = load_data(k=2)
    kf = cross_validation.KFold(len(x), n_fold)
    scaler = preprocessing.StandardScaler()
    acc, prec, recall = [], [], []
    # # debug
    c_set = [0.25, 0.25, 1, 1, 0.5]
    for train, test in kf:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
#         c_star = choose_c(x_train, y_train)
        c_star = c_set[k_out]
        print '=========c*:{}'.format(c_star)
        scaler.fit(x_train)
        clf = svm.LinearSVC(C=c_star)
        clf.fit(scaler.transform(x_train), y_train)
        y_pred = clf.predict(scaler.transform(x_test))
        acc.append(accuracy_score(y_test, y_pred))
        print acc
        prec.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
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
    
        
