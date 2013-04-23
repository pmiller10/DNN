from dnn import DNN
from sklearn import datasets
import sys
sys.path.append("../handwriting_classification")
from preprocess import Preprocess

def score(preds, targets):
    preds - targets
    dif = [(p - targets[i])for i,p in enumerate(preds)]
        

boston = datasets.load_boston()
matrix = Preprocess.to_matrix(list(boston.data))
matrix = Preprocess.scale(matrix)
matrix = list(matrix)
target = list(boston.target)
layers = [13,5,1]

dnn = DNN(matrix, target, layers, hidden_layer="SigmoidLayer", final_layer="LinearLayer", compression_epochs=150, smoothing_epochs=0, bias=True)
dnn.fit()

for i in range(10):
    d = matrix[i]
    t = target[i]
    pred = dnn.predict(d)
    #print "Prediction: {0} for data {1} target: {2}".format(pred, d, t)
    print "Prediction: {0} for target: {1}".format(pred, t)
