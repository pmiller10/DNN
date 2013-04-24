from dnn import DNN
from sklearn import datasets
import sys
sys.path.append("../handwriting_classification")
from preprocess import Preprocess
import math

""" Mean Root Squared Error
Accepts a list of predictions and a list of targets """
def mrse(preds, targets):
    dif = [(math.sqrt((p - targets[i]) ** 2)) for i,p in enumerate(preds)]
    score = float(sum(dif))/len(targets)
    return score

""" Root Mean Squared Error
Accepts a list of predictions and a list of targets """
def rmse(preds, targets):
    dif = [((p - targets[i]) ** 2) for i,p in enumerate(preds)]
    mean = float(sum(dif))/len(targets)
    root = math.sqrt(mean)
    return root

""" Average of a list """
def avg(array):
    return float(sum(array))/len(array)
        

boston = datasets.load_boston()
matrix = Preprocess.to_matrix(list(boston.data))
matrix = Preprocess.scale(matrix)
matrix = list(matrix)
target = list(boston.target)
layers = [13,6,1]

dnn = DNN(matrix, target, layers, hidden_layer="TanhLayer", final_layer="LinearLayer", compression_epochs=20, smoothing_epochs=0, bias=True)
full = dnn.fit()
print full
#preds = [dnn.predict(d)[0] for d in matrix]
preds = [full.activate(d)[0] for d in matrix]

print "mrse preds {0}".format(mrse(preds, target))
print "rmse preds {0}".format(rmse(preds, target))

#mean = avg(target)
#mean = [mean for i in range(len(target))]
#print "mrse mean {0}".format(mrse(mean, target))
#print "rmse mean {0}".format(rmse(mean, target))

for i in range(10):
    d = matrix[i]
    t = target[i]
    pred = full.activate(d)
    #print "Prediction: {0} for data {1} target: {2}".format(pred, d, t)
    print "Prediction: {0} for target: {1}".format(pred, t)

print "\n"

for i in range(10):
    d = matrix[i]
    t = target[i]
    pred = dnn.predict(d)
    #print "Prediction: {0} for data {1} target: {2}".format(pred, d, t)
    print "Prediction: {0} for target: {1}".format(pred, t)
