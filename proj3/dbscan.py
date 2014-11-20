from sklearn.cluster import DBSCAN
from sklearn import metrics
from util import load_data
import matplotlib.pyplot as pl
import numpy as np
    
def cluster():
    eps_set = 0.5 * np.arange(1, 7)
    npt_set = np.arange(1, 6)
    scores = []
    global res
    res = []
    for eps in eps_set:
        for npt in npt_set:
            est = DBSCAN(eps=eps, min_samples=npt)
            est.fit(x)
            ari = metrics.adjusted_rand_score(y, est.labels_)
            scores.append(ari)
            n_noise = len([ l for l in est.labels_ if l == -1])
            res.append((ari, np.max(est.labels_) + 1 , n_noise))
            print ari
    max_score = np.max(scores)
    max_idx = scores.index(max_score)
    max_eps = eps_set[max_idx / len(npt_set)]
    max_npt = npt_set[max_idx % len(npt_set)]
    print max_score, max_eps, max_npt
    scores = np.array(scores).reshape(len(eps_set), len(npt_set))
    pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
    pl.colorbar()
    pl.xticks(np.arange(len(npt_set)), npt_set)
    pl.yticks(np.arange(len(eps_set)), eps_set)
    pl.ylabel('eps')
    pl.xlabel('min_samples')
    pl.show()
    
def test():
    global est
    est = DBSCAN(eps=1, min_samples=1)
    est.fit(x)
    print est.labels_
    ari = metrics.adjusted_rand_score(y, est.labels_)
    print ari
    
if __name__ == "__main__":
    x, y = load_data(k=2)
    cluster()
#     test()
