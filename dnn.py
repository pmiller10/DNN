from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer

class DNN(object):

    def __init__(self, data, targets, layers=[], compression_layer="SigmoidLayer", final_layer="SigmoidLayer", compression_epochs=100, smoothing_epochs=10, verbose=False):
        self.layers = layers
        self.data = data
        self.targets = targets
        self.compression_epochs = compression_epochs
        self.smoothing_epochs = smoothing_epochs
        self.verbose = verbose

        # compression layer
        if compression_layer == "SigmoidLayer":
            self.compression_layer = SigmoidLayer
        elif compression_layer == "LinearLayer":
            self.compression_layer = LinearLayer
        elif compression_layer == "TanhLayer":
            self.compression_layer = TanhLayer
        elif compression_layer == "SoftmaxLayer":
            self.compression_layer = SoftmaxLayer
        else:
            raise Exception("compression_layer must be either: 'LinearLayer', 'SoftmaxLayer' or 'TanhLayer''")

        # final layer
        if final_layer == "SigmoidLayer":
            self.final_layer = SigmoidLayer
        elif final_layer == "LinearLayer":
            self.final_layer = LinearLayer
        elif final_layer == "TanhLayer":
            self.final_layer = TanhLayer
        elif final_layer == "SoftmaxLayer":
            self.final_layer = SoftmaxLayer
        else:
            raise Exception("final_layer must be either: 'LinearLayer', 'SoftmaxLayer' or 'TanhLayer''")

    def predict(self, data):
        if self.nn:
            return self.nn.activate(data)
        else:
            raise Exception("You must run DNN().fit() before you can use predict()") # TODO initialize this with an initial neural network
    def fit(self):
        training_data = self.data
        mid_layers = self.layers[1:]
        mid_layers.pop()
        params = []
        for i,current in enumerate(mid_layers):
            prior = self.layers[i]
            ds = SupervisedDataSet(prior,prior)
            for d in training_data: ds.addSample(d, d)
            print training_data
    
            temp_nn = buildNetwork(prior, current, prior, bias=False, hiddenclass=self.compression_layer, outclass=self.compression_layer)
            trainer = BackpropTrainer(temp_nn, dataset=ds, momentum=0.1, verbose=self.verbose, weightdecay=0.01)
            trainer.trainEpochs(self.compression_epochs)
            params += self.strip_params(temp_nn, prior, current)

            """ Compress the training data """
            layer = buildNetwork(prior, current, bias=False, outclass=self.compression_layer) # TODO you haven't assigned the learned params to this network! So it doesn't work at all!
            layer.params [:] = self.strip_params(temp_nn, prior, current)
            new_data = []
            for d in training_data:
                new_data.append(layer.activate(d))
            training_data = new_data

        """ Train the softmax layer """
        softmax_layer = buildNetwork(self.layers[-2], self.layers[-1], bias=False, outclass=self.final_layer)
        ds = SupervisedDataSet(self.layers[-2], self.layers[-1])
        print training_data
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
        return buildNetwork(self.layers[0], self.layers[1], self.layers[2], bias=False, hiddenclass=self.compression_layer, outclass=self.final_layer)

    """ Accesses the relevant params from the current NN """
    def strip_params(self, temp_nn, prior, current):
        count = prior * current
        p = temp_nn.params[:count]
        return list(p)


def test():
    a = [0,0,1,1]
    b = [1,1,0,0]
    c = [1,1,1,1]
    d = [0,0,0,0]
    data = [a,b,c,d]
    targets = [0.5,0.5,1,0]
    #targets = [1,1,0,0]
    layers = [4,2,1]
    dnn = DNN(data, targets, layers, compression_layer="TanhLayer", compression_epochs=100, smoothing_epochs=10)
    dnn.fit()
    for d in data:
        print dnn.predict(d)
        #print dnn.nn.activate(d)

test()
