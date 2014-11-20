from sklearn.cluster import SpectralClustering
from sklearn import metrics
from util import load_data
import matplotlib.pyplot as pl
import math
import numpy as np

def rbf(max_k):
    gamma_set = [math.pow(10, i) for i in xrange(-5, 1)]
    a_score, idx = [[] for i in xrange(len(gamma_set))], []
    for k in xrange(2, max_k + 1):
        print 'k={}'.format(k)
        for i, gamma in enumerate(gamma_set):
            est = SpectralClustering(n_clusters=k, affinity='rbf', gamma=gamma)
            est.fit(x)
            ari = metrics.adjusted_rand_score(y, est.labels_)
            a_score[i].append(ari)
        idx.append(k)
    for i in xrange(len(gamma_set)):
        print gamma_set[i]
        print np.max(a_score[i])
        pl.plot(idx, a_score[i], label='gamma={}'.format(gamma_set[i]))
        
    pl.legend(loc=4,prop={'size':12})
    pl.xlabel('# of clusters')
    pl.ylabel('ARI')
    pl.show()
    
def eval_k(max_k):
    a_score, idx = [], []
    for k in xrange(2, max_k + 1):
        print 'k={}'.format(k)
        est = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
#         est = SpectralClustering(n_clusters=k, affinity='rbf', gamma=0.00001)
        est.fit(x)
        ari = metrics.adjusted_rand_score(y, est.labels_)
        print ari
        a_score.append(ari)
        idx.append(k)
    pl.plot(idx, a_score)
    pl.xlabel('# of clusters')
    pl.ylabel('ARI')
    pl.show()
#         break

if __name__ == "__main__":
    x, y = load_data(k=2)
    rbf(10)
#     eval_k(15)
