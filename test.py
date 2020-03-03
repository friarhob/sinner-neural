from sinner import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork(layersSize=[2,4,5,1],learningRate=1)
    inputs = [[0,0],
              [0,1],
              [1,0],
              [1,1]]
    outputs = [[1],[0],[0],[1]]
    print(nn.train(inputs,outputs,epochs=10000))
    for inputRow in inputs:
        print(str(inputRow)+" "+str(nn.eval(inputRow)))
    print(nn)

