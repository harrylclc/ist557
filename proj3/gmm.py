from sklearn.mixture import GMM
from sklearn import metrics
from util import load_data
import matplotlib.pyplot as pl
import numpy as np

def rand_test(n_iter=5, k=8, verbose=True):
    ari, s = [], []
    for i in xrange(n_iter):
        if verbose:
            print 'iter: {}'.format(i)
        est = GMM(n_components=k, covariance_type='full')
        est.fit(x)
        y_pred = est.predict(x)
        ari_v = metrics.adjusted_rand_score(y, y_pred)
        s_v = metrics.silhouette_score(x, y_pred, metric='euclidean')
        if verbose:
            print ari_v, s_v
        ari.append(ari_v)
        s.append(s_v)
    return np.mean(ari), np.mean(s)

def eval_k(max_k=10):
    a_score, s_score, idx = [], [], []
    for k in xrange(2, max_k + 1):
        print 'k={}'.format(k)
        ari, s = rand_test(k=k, verbose=False)
        print ari, s
        a_score.append(ari)
        s_score.append(s)
        idx.append(k)
    print 'max ari: {}'.format(a_score.index(np.max(a_score)))
    print 'max s: {}'.format(s_score.index(np.max(s_score)))
    # plot
    pl.plot(idx, a_score, '-o', label='adjusted Rand index')
    pl.plot(idx, s_score, '-o', label='Silhouette Coefficient')
    pl.legend()
    pl.ylabel('Result')
    pl.xlabel('K')
    pl.show()
    

if __name__ == "__main__":
    x, y = load_data(2)
    eval_k(max_k=10)
