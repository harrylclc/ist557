import numpy as np
import matplotlib.pyplot as plt

data = []
with open('../data/rf_trees') as f:
    for line in f:
        l = map(lambda x: float(x), line.strip().split(','))
        data.append(l)
data = np.array(data)
baseline = [0.9250, 0.7212, 0.7187, 0.7200]
labels = ["Accuracy", "Precision", "Recall", "F1"]

for i in xrange(4):
    d = data[:, i]
    idx = np.arange(1, len(d) + 1)
    ref = [baseline[i] for j in xrange(len(d))]
    plt.clf()
    plt.plot(idx, d, label='Random Forest')
    plt.plot(idx, ref, label='Decision Tree')
    plt.xlabel("Number of trees")
    plt.ylabel(labels[i])
    plt.legend(numpoints=1, loc=4)
    plt.savefig('figs/compare_{}.png'.format(labels[i]))
#     plt.show()
    plt.close()
    
        
