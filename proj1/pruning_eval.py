from util import load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
n_fold = 5

def pruning_cross_validation(x, y, cv):
    min_m , max_m, delta_m = 1, 1000, 10
    size = (max_m - min_m) / delta_m + 1
    print size
    a_score = [[] for i in xrange(size)]
    p_score = [[] for i in xrange(size)]
    r_score = [[] for i in xrange(size)]
    k_in = 0
    for train, test in cv:
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        m = min_m
        while 1:
            print "Out:{} In:{} iter:{}".format(k_out, k_in, m)
            clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=m + 1)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_test, y_pred) 
            print 'accuracy: {}'.format(acc)
            m_idx = (m - min_m) / delta_m
            a_score[m_idx].append(acc)
            p_score[m_idx].append(precision_score(y_test, y_pred))
            r_score[m_idx].append(recall_score(y_test, y_pred))
            m += delta_m
            if m > max_m:
                break
        k_in += 1
    max_val, max_id = -1, -1
    for i in xrange(len(a_score)):
        acc = np.mean(a_score[i])
        if acc > max_val:
            max_val = acc
            max_id = i
        a_score[i] = acc
    return max_id * delta_m + min_m

def main():
    global k_out
    k_out = 0
    x, y = load_data(k=2)
    kf_out = cross_validation.KFold(len(x), n_fold)
    a_score, p_score, r_score = [], [], []
    for train_out, test_out in kf_out:
        x_train_out, x_test_out, y_train_out, y_test_out = x[train_out] , x[test_out] , y[train_out] , y[test_out]
        kf = cross_validation.KFold(len(x_train_out), n_fold)
        m_opt = pruning_cross_validation(x_train_out, y_train_out, kf)
        clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=m_opt + 1)
        print '=========m_opt:{}'.format(m_opt)
        clf.fit(x_train_out, y_train_out)
        y_pred = clf.predict(x_test_out)
        a_score.append(accuracy_score(y_test_out, y_pred))
        p_score.append(precision_score(y_test_out, y_pred))
        r_score.append(recall_score(y_test_out, y_pred))
        k_out += 1
    a = np.mean(a_score)
    p = np.mean(p_score)
    r = np.mean(r_score)
    f = 2 * p * r / (p + r)
    print 'precision: {}'.format(p)
    print "recall: {}".format(r)
    print "f1: {}".format(f)
    print "accuracy: {}".format(a)
                    
if __name__ == "__main__":
    main()
