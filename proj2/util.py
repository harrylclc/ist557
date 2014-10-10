from sklearn.datasets import load_iris, make_classification, load_svmlight_file
from sklearn import cross_validation
def load_data(k=0):
    print 'loading'
    fname = '../data/art_feature_4week'
    if k == 0:
        iris = load_iris()
        x = iris.data
        y = iris.target
    elif k == 1:
        toy_data = make_classification(n_samples=500, n_features=14)
        x = toy_data[0]
        y = toy_data[1]
    elif k == 2:
        x, y = load_svmlight_file(fname)
        x = x.toarray()
    elif k == 3:
        x, y = load_svmlight_file(fname)
        x = x.toarray()
        skf = cross_validation.StratifiedKFold(y, 10)
        for train, test in skf:
            print train, test
            x, y = x[test], y[test]
            break
    print 'done'
    return x, y
