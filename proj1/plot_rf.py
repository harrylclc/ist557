import numpy as np
import matplotlib.pyplot as plt

data = []
with open('res/rf_trees') as f:
    for line in f:
        l = map(lambda x: float(x), line.strip().split(','))
        data.append(l)
data = np.array(data)
baseline = [0.9378, 0.7732, 0.7624, 0.7678]
labels = ["Accuracy", "Precision", "Recall", "F1"]

for i in xrange(4):
    d = data[:, i]
    idx = np.arange(1, len(d) + 1) * 10
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
    
        
