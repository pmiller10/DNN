from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import SoftmaxLayer

class DNN(object):

    def __init__(self, data, targets, layers=[], layer_type="SigmoidLayer", compression_epochs=100, smoothing_epochs=10, verbose=False):
        self.layers = layers
        self.data = data
        self.targets = targets
        self.compression_epochs = compression_epochs
        self.smoothing_epochs = smoothing_epochs
        self.verbose = verbose
        if layer_type == "SigmoidLayer":
            self.layer_type = SigmoidLayer
        elif layer_type == "LinearLayer":
            self.layer_type = LinearLayer
        elif layer_type == "TanhLayer":
            self.layer_type = TanhLayer
        elif layer_type == "SoftmaxLayer":
            self.layer_type = SoftmaxLayer
        else:
            raise Exception("layer_type must be either: 'LinearLayer', 'SoftmaxLayer' or 'TanhLayer''")

    def fit(self):
        training_data = self.data
        mid_layers = self.layers[1:]
        mid_layers.pop()
        params = []
        for i,current in enumerate(mid_layers):
            prior = self.layers[i]
            ds = SupervisedDataSet(prior,prior)
            for d in training_data: ds.addSample(d, d)
    
            temp_nn = buildNetwork(prior, current, prior, bias=False, hiddenclass=self.layer_type, outclass=self.layer_type)
            trainer = BackpropTrainer(temp_nn, dataset=ds, momentum=0.1, verbose=self.verbose, weightdecay=0.01)
            trainer.trainEpochs(self.compression_epochs)
            params += self.strip_params(temp_nn, prior, current)

            """ Compress the training data """
            layer = buildNetwork(prior, current, bias=False)
            new_data = []
            for d in training_data:
                new_data.append(layer.activate(d))
            training_data = new_data

        """ Train the softmax layer """
        softmax_layer = buildNetwork(self.layers[-2], self.layers[-1], bias=False)
        ds = SupervisedDataSet(self.layers[-2], self.layers[-1])
        for i,d in enumerate(training_data):
            target = self.targets[i]
            ds.addSample(d, target)
        trainer = BackpropTrainer(softmax_layer, dataset=ds, momentum=0.1, verbose=self.verbose, weightdecay=0.01)
        trainer.trainEpochs(self.compression_epochs)
        params += self.strip_params(softmax_layer, prior, current)

        real_nn = self.real_nn()
        real_nn.params[:] = params

        """ smoothing """
        ds = SupervisedDataSet(self.layers[0], self.layers[-1])
        real_nn
        for i,d in enumerate(self.data):
            target = self.targets[i]
            ds.addSample(d, target)
        trainer = BackpropTrainer(real_nn, dataset=ds, momentum=0.1, verbose=self.verbose, weightdecay=0.01)
        trainer.trainEpochs(self.smoothing_epochs)
        self.nn = real_nn
        
    def real_nn(self):
        return buildNetwork(self.layers[0], self.layers[1], self.layers[2], bias=False, hiddenclass=self.layer_type, outclass=self.layer_type)

    """ Accesses the relevant params from the current NN """
    def strip_params(self, temp_nn, prior, current):
        count = prior * current
        p = temp_nn.params[:count]
        return list(p)


def test():
    a = [0,0,1,1]
    b = [1,1,0,1]
    c = [1,1,1,1]
    d = [0,0,0,0]
    data = [a,b,c,d]
    targets = [0.5,0.5,1,0]
    #targets = [1,1,0,0]
    layers = [4,2,1]
    dnn = DNN(data, targets, layers, layer_type="LinearLayer", compression_epochs=10, smoothing_epochs=50)
    dnn.fit()
    print dnn.nn.activate([0,0,0,0])
    print dnn.nn.activate([0,0,1,1])
    print dnn.nn.activate([1,1,0,0])
    print dnn.nn.activate([1,1,1,1])

test()
