from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer, BiasUnit, FeedForwardNetwork, FullConnection
import numpy
import copy



class Layer():

    SIGMOID = SigmoidLayer
    LINEAR = LinearLayer
    TANH = TanhLayer
    SOFTMAX = SoftmaxLayer


class AutoEncoder(object):

    # TODO still need to add one more layer than you actually want because this is training the softmax
    # need to create a DNNClassifier class in addition to the Regressor class
    def __init__(self, supervised, unsupervised, targets, layers=[], hidden_layer="SigmoidLayer", final_layer="SigmoidLayer", compression_epochs=100, verbose=True, bias=True, autoencoding_only=True, dropout_on=True):
        self.layers = layers
        self.supervised = supervised
        self.unsupervised = unsupervised
        self.targets = targets
        self.compression_epochs = compression_epochs
        self.verbose = verbose
        self.bias = bias
        self.autoencoding_only = autoencoding_only
        self.nn = []
        self.dropout_on = dropout_on

        methods = dir(Layer)
        methods.remove('__doc__')
        methods.remove('__module__')

        # compression layer
        assert hidden_layer in dir(Layer), "hidden_layer must be in {0}".format(methods)
        self.hidden_layer = getattr(Layer, hidden_layer)

        # final layer
        assert final_layer in dir(Layer), "final_layer must be in {0}".format(methods)
        self.final_layer = getattr(Layer, final_layer)

    def predict(self, data):
        if not self.nn: raise Exception("You must run ._train() before you can predict")
        for nn in self.nn:
            data = nn.activate(data)
        return data

    def fit(self):
        autoencoder, _, _, _ = self._train()
        autoencoder.sortModules()
        return autoencoder

    def _train(self):
        hidden_layers = []
        bias_layers = []
        compressed_data = copy.copy(self.unsupervised) # it isn't compressed at this point, but will be later on
        compressed_supervised = self.supervised

        mid_layers = self.layers[1:-1] # remove the first and last
        for i,current in enumerate(mid_layers):
            prior = self.layers[i] # This accesses the layer before the "current" one, since the indexing in mid_layers and self.layers is offset by 1

            # build the NN with a bottleneck
            bottleneck = FeedForwardNetwork()
            in_layer = LinearLayer(prior)
            hidden_layer = self.hidden_layer(current)
            out_layer = self.hidden_layer(prior)
            bottleneck.addInputModule(in_layer)
            bottleneck.addModule(hidden_layer)
            bottleneck.addOutputModule(out_layer)
            in_to_hidden = FullConnection(in_layer, hidden_layer)
            hidden_to_out = FullConnection(hidden_layer, out_layer)
            bottleneck.addConnection(in_to_hidden)
            bottleneck.addConnection(hidden_to_out)
            if self.bias:
                bias1 = BiasUnit()
                bias2 = BiasUnit()
                bottleneck.addModule(bias1)
                bottleneck.addModule(bias2)
                bias_in = FullConnection(bias1, hidden_layer)
                bias_hidden = FullConnection(bias2, out_layer)
                bottleneck.addConnection(bias_in)
                bottleneck.addConnection(bias_hidden)
            bottleneck.sortModules()

            # train the bottleneck
            print "\n...training for layer ", prior, " to ", current
            ds = SupervisedDataSet(prior,prior)
            if self.dropout_on:
                noisy_data, originals = self.dropout(compressed_data, noise=0.2, bag=1)
                for i,n in enumerate(noisy_data):
                    original = originals[i]
                    ds.addSample(n, original)
            else:
                for d in (compressed_data): ds.addSample(d, d)
            trainer = BackpropTrainer(bottleneck, dataset=ds, learningrate=0.001, momentum=0.05, verbose=self.verbose, weightdecay=0.05)
            trainer.trainEpochs(self.compression_epochs)
            if self.verbose: print "...data:\n...", compressed_data[0][:8], "\nreconstructed to:\n...", bottleneck.activate(compressed_data[0])[:8]

            hidden_layers.append(in_to_hidden)
            if self.bias: bias_layers.append(bias_in)

            # use the params from the bottleneck to compress the training data
            compressor = FeedForwardNetwork()
            compressor.addInputModule(in_layer)
            compressor.addOutputModule(hidden_layer) # use the hidden layer from above
            compressor.addConnection(in_to_hidden)
            compressor.sortModules()
            compressed_data = [compressor.activate(d) for d in compressed_data]
            compressed_supervised = [compressor.activate(d) for d in compressed_supervised]

            self.nn.append(compressor)

        # Train the softmax layer
        print "\n...training for softmax layer "
        softmax = FeedForwardNetwork()
        in_layer = LinearLayer(self.layers[-2])
        out_layer = self.final_layer(self.layers[-1])
        softmax.addInputModule(in_layer)
        softmax.addOutputModule(out_layer)
        in_to_out = FullConnection(in_layer, out_layer)
        softmax.addConnection(in_to_out)
        if self.bias:
            bias = BiasUnit()
            softmax.addModule(bias)
            bias_in = FullConnection(bias, out_layer)
            softmax.addConnection(bias_in)
        softmax.sortModules()

        # see if it's for classification or regression
        if self.final_layer == SoftmaxLayer:
            print "...training for a softmax network"
            ds = ClassificationDataSet(self.layers[-2], 1)
        else:
            print "...training for a regression network"
            ds = SupervisedDataSet(self.layers[-2], self.layers[-1])
        bag = 1
        noisy_data, _ = self.dropout(compressed_supervised, noise=0.5, bag=bag)
        bagged_targets = []
        for t in self.targets:
            for b in range(bag):
                bagged_targets.append(t)
        for i,d in enumerate(noisy_data):
            target = bagged_targets[i]
            ds.addSample(d, target)

        # see if it's for classification or regression
        if self.final_layer == SoftmaxLayer:
            ds._convertToOneOfMany()

        # TODO make these configurable
        trainer = BackpropTrainer(softmax, dataset=ds, learningrate=0.001, momentum=0.05, verbose=self.verbose, weightdecay=0.05)
        trainer.trainEpochs(self.compression_epochs)
        self.nn.append(softmax)
        hidden_layers.append(in_to_out)
        if self.bias: bias_layers.append(bias_in)

        # Recreate the whole thing
        # connect the first two
        autoencoder = FeedForwardNetwork()
        first_layer = hidden_layers[0].inmod
        next_layer = hidden_layers[0].outmod
        autoencoder.addInputModule(first_layer)
        connection = FullConnection(first_layer, next_layer)
        connection.params[:] = hidden_layers[0].params
        autoencoder.addConnection(connection)

        # decide whether this should be the output layer or not
        if self.autoencoding_only and (len(self.layers) <= 3): # TODO change this to 2 when you aren't using the softmax above
            autoencoder.addOutputModule(next_layer)
        else:
            autoencoder.addModule(next_layer)
        if self.bias:
            bias = bias_layers[0]
            bias_unit = bias.inmod
            autoencoder.addModule(bias_unit)
            connection = FullConnection(bias_unit, next_layer)
            connection.params[:] = bias.params
            autoencoder.addConnection(connection)

        # connect the middle layers
        for i,h in enumerate(hidden_layers[1:-1]):
            new_next_layer = h.outmod

            # decide whether this should be the output layer or not
            if self.autoencoding_only and i == (len(hidden_layers) - 3):
                autoencoder.addOutputModule(new_next_layer)
            else:
                autoencoder.addModule(new_next_layer)
            connection = FullConnection(next_layer, new_next_layer)
            connection.params[:] = h.params
            autoencoder.addConnection(connection)
            next_layer = new_next_layer

            if self.bias:
                bias = bias_layers[i+1]
                bias_unit = bias.inmod
                autoencoder.addModule(bias_unit)
                connection = FullConnection(bias_unit, next_layer)
                connection.params[:] = bias.params
                autoencoder.addConnection(connection)

        return autoencoder, hidden_layers, next_layer, bias_layers

    def dropout(self, data, noise=0., bag=1, debug=False):
        if bag < 1:
            raise Exception("bag must be 1 or greater")
        length = len(data[0])
        zeros = round(length * noise)
        ones = length - zeros
        zeros = numpy.zeros(zeros)
        ones = numpy.ones(ones)
        merged = numpy.concatenate((zeros, ones), axis=1)
        dropped = []
        originals = []
        bag = range(bag) # increase by this amount
        for d in data:
            for b in bag:
                numpy.random.shuffle(merged)
                dropped.append(merged * d)
                originals.append(d)
        if self.verbose:
            print "...number of data: ", len(data)
            print "...number of bagged data: ", len(dropped)
            print "...data: ", data[0][:10]
            print "...noisy data: ", dropped[0][:10]
        return dropped, originals


class DNNRegressor(AutoEncoder):

    def fit(self):
        autoencoder, hidden_layers, next_layer, bias_layers = self._train()
        with_top_layer = self._top_layer(autoencoder, hidden_layers, next_layer, bias_layers)
        new = buildNetwork(*self.layers, hiddenclass=self.hidden_layer, outclass=self.final_layer)
        new.params[:] = with_top_layer.params
        return new

    def _top_layer(self, autoencoder, hidden_layers, next_layer, bias_layers):
        # connect 2nd to last and last
        last_layer = hidden_layers[-1].outmod
        autoencoder.addOutputModule(last_layer)
        connection = FullConnection(next_layer, last_layer)
        connection.params[:] = hidden_layers[-1].params
        autoencoder.addConnection(connection)
        if self.bias:
            bias = bias_layers[-1]
            bias_unit = bias.inmod
            autoencoder.addModule(bias_unit)
            connection = FullConnection(bias_unit, last_layer)
            connection.params[:] = bias.params
            autoencoder.addConnection(connection)

        autoencoder.sortModules()
        return autoencoder
