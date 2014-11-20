from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from util import load_data
import matplotlib.pyplot as plt
import numpy as np

def calc_distance(n_clusters, labels):
    clusters = [[] for i in xrange(n_clusters)]
    for idx, l in enumerate(labels):
        clusters[l].append(idx)
    min_distances, max_distances, mean_distances = [] , [] , []
    for i in xrange(n_clusters):
        for j in xrange(i + 1, n_clusters):
            c_i = x[clusters[i]]
            c_j = x[clusters[j]]
            distances = metrics.pairwise.euclidean_distances(c_i, c_j)
#             distances = []
#             for x_i in c_i:
#                 for x_j in c_j:
#                     d = np.linalg.norm(x_i - x_j)
#                     distances.append(d)
            min_d = np.min(distances)
            max_d = np.max(distances)
            mean_d = np.mean(distances)
            min_distances.append(min_d)
            max_distances.append(max_d)
            mean_distances.append(mean_d)
    return [np.min(min_distances), np.min(max_distances), np.min(mean_distances)]

def eval_dist(linkage='ward'):
    a_score = []
    idx = []
    d = [[] for i in xrange(3)]
    for k in xrange(2, 50 + 1):
        print 'k={}'.format(k)
        est = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        est.fit(x)
        ari_v = metrics.adjusted_rand_score(y, est.labels_)
        ds = calc_distance(k, est.labels_)
        for i in xrange(3):
            d[i].append(ds[i])
        print ari_v
        a_score.append(ari_v)
        idx.append(k)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(idx, a_score)
#     plt.xlim(0, 220)
    axes[0].set_ylim(ymin=0)
    axes[0].set_ylabel('ARI')
    axes[0].set_xlabel('# of clusters')
#     plt.savefig('figs/hc_ari.png')
#     plt.show()
#     plt.close()
    labels = ['Minimum', 'Maximum', 'Average']
#     for i in xrange(3):
#         axes[1].plot(idx, d[i], label=labels[i])
    axes[1].plot(idx, d[1])
    axes[1].legend()
    axes[1].set_ylabel('distance')
    axes[1].set_xlabel('# of clusters')
#     plt.savefig('figs/hc_distance.png')
    plt.show()
#     plt.close()
    
def cluster():
    linkages = ['ward', 'complete', 'average']
    labels = ['ward', 'Complete-link', 'Average']
    for i, linkage in enumerate(linkages):
        a_score, idx = [], []
        for k in xrange(2, len(x) + 1):
            print 'k={}'.format(k)
            est = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            est.fit(x)
            ari_v = metrics.adjusted_rand_score(y, est.labels_)
#             print ari_v
            a_score.append(ari_v)
            idx.append(k)
        max_idx = a_score.index(np.max(a_score))
        print 'max_idx: {}'.format(max_idx + 2)
        print 'max: {}'.format(np.max(a_score))
        plt.plot(idx, a_score, label=labels[i], linewidth=1.5)
    plt.xlabel('# of clusters')
    plt.ylabel('ARI')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x, y = load_data(k=2)
#     cluster()
    eval_dist(linkage='complete')
