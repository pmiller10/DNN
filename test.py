from dnn import AutoEncoder, DNNRegressor

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

    layers = [4,2,1]
    dnn = AutoEncoder(data, data, targets, layers, hidden_layer="TANH", final_layer="TANH", compression_epochs=50, bias=True, autoencoding_only=True)
    #dnn = DNNRegressor(data, targets, layers, hidden_layer="TanhLayer", final_layer="TanhLayer", compression_epochs=50, bias=True, autoencoding_only=False)
    dnn = dnn.fit()
    data.append([0.9, 0.8, 0, 0.1])
    print "\n-----"
    for d in data:
        print dnn.activate(d)

if __name__ == "__main__":
    test()
