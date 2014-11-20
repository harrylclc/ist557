from sklearn.datasets import load_iris
from sklearn.preprocessing  import scale, StandardScaler
import numpy as np
def load_data(k=0, scale=True):
    print 'loading'
    if k == 0:
        iris = load_iris()
        x = iris.data
        y = iris.target
    elif k == 1:
        x, y = [], []
        labels = {'CYT':0, 'NUC':1, 'MIT':2, 'ME3':3, 'ME2':4, 'ME1':5, 'EXC':6, 'VAC':7, 'POX':8, 'ERL':9}
        with open('../data/yeast.data') as f:
            for line in f:
                s = line.split()
                x.append([float(v) for v in s[1:-1]])
                y.append(labels[s[-1]])
        x = np.array(x)
        y = np.array(y)
    elif k == 2:
        x, y = [], []
        with open('../data/glass.data') as f:
            for line in f:
                s = line.strip().split(',')
                x.append([float(v) for v in s[1:-1]])
                y.append(int(s[-1]))
        x = np.array(x)
        y = np.array(y)
    elif k==3:
        x, y = [], []
        with open('../data/wine.data') as f:
            for line in f:
                s = line.strip().split(',')
                x.append([float(v) for v in s[1:]])
                y.append(int(s[0]))
        x = np.array(x)
        y = np.array(y)
    if scale:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
    print 'done'
    return x, y

if __name__ == "__main__":
    x, y = load_data(2)
    print x, y
