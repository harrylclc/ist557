from util import load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
n_fold = 5

def main():
    x, y = load_data(k=2)
    print len(x)
    kf = cross_validation.KFold(len(x), n_fold)
    max_m = min(5000, int(len(x) * (n_fold - 1) / n_fold) - 1)
    acc_score = [[] for i in xrange(max_m)]
    p_score = [[] for i in xrange(max_m)]
    r_score = [[] for i in xrange(max_m)]
    for train, test in kf:
        print len(train)
        x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
        m = 1
        
        while 1: 
            print "iter: {}".format(m)
            clf = DecisionTreeClassifier(max_leaf_nodes=m + 1)
            clf.fit(x_train, y_train)
#             p = clf.predict(x_train)
#             print accuracy_score(y_train, p)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            acc_score[m - 1].append(acc)
            p_score[m - 1].append(precision_score(y_test, y_pred))
            r_score[m - 1].append(recall_score(y_test, y_pred))
            print 'accuracy: {}'.format(acc)
            m += 1
            if m > max_m:
                break             
#         break
    max_val, max_id = -1, -1
    for i in xrange(len(acc_score)):
        acc = np.mean(acc_score[i])
        if acc > max_val:
            max_val = acc
            max_id = i
        acc_score[i] = acc
        p_score[i] = np.mean(p_score[i])
        r_score[i] = np.mean(r_score[i])
    print acc_score[:10]
    print 'splits:{}'.format(max_id + 1)
    print 'accuracy:{}'.format(max_val)
    print 'p:{}    r:{}'.format(p_score[max_id], r_score[max_id])
    
    plt.clf()
    m_idx = np.arange(len(acc_score)) + 1
    plt.plot(m_idx, acc_score, label='cross_validation')
    plt.plot(max_id + 1, max_val, linestyle='none', marker='o', markeredgecolor='r', markeredgewidth=1, markersize=12, markerfacecolor='none', label='best choice')
    plt.plot((max_id + 1, max_id + 1), (0, max_val), 'k--')
    plt.ylim(ymin=0.9, ymax=0.96)
    plt.xlabel("Number of splits")
    plt.ylabel("Cross validation score")
    plt.legend(numpoints=1, loc=4)
    plt.savefig('effect_of_splits.png')
#     plt.show()

if __name__ == "__main__":
    main()
