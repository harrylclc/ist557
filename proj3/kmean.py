from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import metrics
from util import load_data
import numpy as np

def rand_test(n_iter=5,k=8,verbose=True):
    ari,s = [],[]
    for i in xrange(n_iter):
        print 'iter: {}'.format(i)
        est = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300)
        est.fit(x)
        ari_v =  metrics.adjusted_rand_score(y, est.labels_)
        s_v =  metrics.silhouette_score(x, est.labels_, metric='euclidean')
        if verbose:
            print ari_v, s_v
        ari.append(ari_v)
        s.append(s_v)
    return np.mean(ari),np.mean(s)

def eval_k(max_k=10):
    
    return

if __name__ == "__main__":
    x, y = load_data(1)
    x = scale(x)
    rand_test()
