from sklearn.datasets import load_iris, make_classification, load_svmlight_file
def load_data(k=0):
    print 'loading'
    if k == 0:
        iris = load_iris()
        x = iris.data
        y = iris.target
    elif k == 1:
        toy_data = make_classification(n_samples=50000, n_features=14)
        x = toy_data[0]
        y = toy_data[1]
    elif k == 2:
        fname = '../data/art_feature_4week'
        x, y = load_svmlight_file(fname)
        x = x.toarray()
    print 'done'
    return x, y
