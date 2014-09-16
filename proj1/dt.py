from util import load_data
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

n_fold = 5
criteria = ['gini', 'entropy']
evaluation = 0
plot = 1

def main():
    x, y = load_data(k=2)
    if evaluation:
        kf = cross_validation.KFold(len(x), n_fold)
        for criterion in criteria:
            print 'criterion: {}'.format(criterion)
            acc, prec, recall, node_cnt = [], [], [], []
            clf = DecisionTreeClassifier(criterion=criterion)
            for train, test in kf:
                x_train, x_test, y_train, y_test = x[train] , x[test] , y[train] , y[test]
                clf.fit(x_train, y_train)
                node_cnt.append(clf.tree_.node_count)
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
            print "nodes: {}".format(np.mean(node_cnt))
    
    if plot:
        from sklearn.externals.six import StringIO
        from sklearn import tree
        import pydot
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(x, y)
        print clf.tree_.max_depth
        print clf.tree_.node_count
        dot_data = StringIO()
        tree.export_graphviz(clf, max_depth=3, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        graph.write_pdf("figs/test.pdf") 
#         with open('test.dot', 'w') as f:
#             f = tree.export_graphviz(clf, max_depth=4, out_file=f)
        
if __name__ == "__main__":
    main()
