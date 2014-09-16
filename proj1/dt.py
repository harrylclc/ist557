from util import load_data
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

n_fold = 5
criteria = ['gini', 'entropy']
evaluation = 1
plot = 0

def main():
    x, y = load_data(k=2)
    if evaluation:
        kf = cross_validation.KFold(len(x), n_fold)
        for criterion in criteria:
            print 'criterion: {}'.format(criterion)
            acc, prec, recall = [], [], []
            for train, test in kf:
                x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
                clf = DecisionTreeClassifier(criterion=criterion)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                acc.append(accuracy_score(y_test, y_pred))
                prec.append(precision_score(y_test, y_pred))
                recall.append(recall_score(y_test, y_pred))
            a = np.mean(acc)
            p = np.mean(prec)
            r = np.mean(recall)
            f = 2 * p * r / (p + r)
            print 'precision: {}'.format(p)
            print "recall: {}".format(r)
            print "f1: {}".format(f)
            print "accuracy: {}".format(a)
    
    if plot:
        from sklearn.externals.six import StringIO
        from sklearn import tree
        import pydot
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(x, y)
        dot_data = StringIO()
        tree.export_graphviz(clf, max_depth=3, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        graph.write_pdf("test.pdf") 
#         with open('test.dot', 'w') as f:
#             f = tree.export_graphviz(clf, max_depth=4, out_file=f)
        
if __name__ == "__main__":
    main()
