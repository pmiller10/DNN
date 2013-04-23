from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer

class DNN(object):

    def __init__(self, data, targets, layers=[], hidden_layer="SigmoidLayer", final_layer="SigmoidLayer", compression_epochs=100, smoothing_epochs=10, verbose=False, bias=True):
        self.layers = layers
        self.data = data
        self.targets = targets
        self.compression_epochs = compression_epochs
        self.smoothing_epochs = smoothing_epochs
        self.verbose = verbose
        self.bias = bias
        self.nn = []

        # compression layer
        if hidden_layer == "SigmoidLayer":
            self.hidden_layer = SigmoidLayer
        elif hidden_layer == "LinearLayer":
            self.hidden_layer = LinearLayer
        elif hidden_layer == "TanhLayer":
            self.hidden_layer = TanhLayer
        elif hidden_layer == "SoftmaxLayer":
            self.hidden_layer = SoftmaxLayer
        else:
            raise Exception("hidden_layer must be either: 'LinearLayer', 'SoftmaxLayer', 'SigmoidLayer', or 'TanhLayer'")

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
            raise Exception("final_layer must be either: 'LinearLayer', 'SoftmaxLayer', 'SigmoidLayer', or 'TanhLayer'")

    def predict(self, data):
        for nn in self.nn:
            data = nn.activate(data)
        return data
    #def predict(self, data):
        #if self.nn:
            #return self.nn.activate(data)
        #else:
            #raise Exception("You must run DNN().fit() before you can use predict()") # TODO initialize this with an initial neural network

    def fit(self):
        compressed_data = self.data # NOTE: it isn't compressed at this point, but will be later on
        mid_layers = self.layers[1:] # remove the first
        mid_layers.pop() # remove the last to get only the layers in the middle
        params = []
        for i,current in enumerate(mid_layers):
            """ build the NN with a bottleneck """
            prior = self.layers[i] # This accesses the layer before the "current" one, since the indexing in mid_layers and self.layers is offset by 1
            ds = SupervisedDataSet(prior,prior)
            for d in compressed_data: ds.addSample(d, d)
            #print "Compressed data at stage {0} {1}".format(i, compressed_data)
            bottleneck = buildNetwork(prior, current, prior, bias=self.bias, hiddenclass=self.hidden_layer, outclass=self.hidden_layer)
            trainer = BackpropTrainer(bottleneck, dataset=ds, momentum=0.1, verbose=self.verbose, weightdecay=0.01)
            trainer.trainEpochs(self.compression_epochs)
            compression_params = self.strip_params(bottleneck, prior, current)
            params += compression_params 

            """ Compress the training data """
            layer = buildNetwork(prior, current, bias=self.bias, outclass=self.hidden_layer)
            layer.sortModules()
            layer.params[:] = compression_params
            self.nn.append(layer)
            compressed_data = [layer.activate(d) for d in compressed_data]

        """ Train the softmax layer """
        softmax_layer = buildNetwork(self.layers[-2], self.layers[-1], bias=self.bias, outclass=self.final_layer)
        ds = SupervisedDataSet(self.layers[-2], self.layers[-1])
        for i,d in enumerate(compressed_data):
            target = self.targets[i]
            ds.addSample(d, target)
        trainer = BackpropTrainer(softmax_layer, dataset=ds, momentum=0.1, verbose=self.verbose, weightdecay=0.01)
        trainer.trainEpochs(self.compression_epochs)
        self.nn.append(softmax_layer)
        params += list(softmax_layer.params)

        """ smoothing """
        # right now this isn't being used
        #real_nn = self.real_nn()
        #real_nn.params[:] = params
        #ds = SupervisedDataSet(self.layers[0], self.layers[-1])
        #real_nn
        #for i,d in enumerate(self.data):
        #    target = self.targets[i]
        #    ds.addSample(d, target)
        #trainer = BackpropTrainer(real_nn, dataset=ds, momentum=0.1, verbose=self.verbose, weightdecay=0.01)
        #trainer.trainEpochs(self.smoothing_epochs)
        #self.nn = real_nn # TODO testing only
        
    def real_nn(self):
        nn = buildNetwork(self.layers[0], self.layers[1], self.layers[2], bias=self.bias, hiddenclass=self.hidden_layer, outclass=self.final_layer)
        nn.sortModules()
        return nn

    """ Accesses the relevant params from the current NN """
    def strip_params(self, temp_nn, prior_dims, current_dims):
        count = prior_dims * current_dims
        temp_params = list(temp_nn.params[:count])
        if self.bias:
            out_bias = prior_dims
            hidden_bias = current_dims
            bias_params = temp_nn.params[count:(count+hidden_bias)]
            temp_params += list(bias_params)
        return temp_params


def test():
    data = []
    data.append([0,0,1,1])
    data.append([0,0,1,0.9])
    data.append([0,0,0.9,0.9])
    data.append([0.8,1,0,0])
    data.append([1,1,0.1,0])
    data.append([1,0.9,0,0.2])

    targets = []
    targets.append(0)
    targets.append(0)
    targets.append(0)
    targets.append(1)
    targets.append(1)
    targets.append(1)

    layers = [4,1,1]
    dnn = DNN(data, targets, layers, hidden_layer="TanhLayer", final_layer="TanhLayer", compression_epochs=500, smoothing_epochs=0, bias=True)
    dnn.fit()
    data.append([0.9, 0.8, 0, 0.1])
    for d in data:
        print dnn.predict(d)

test()
