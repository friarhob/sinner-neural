from sinner import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork(layersSize=[2,4,5,1],learningRate=1)
    inputs = [[0,0],
              [0,1],
              [1,0],
              [1,1]]
    outputs = [[0],[1],[1],[0]]

    nn.train(inputs,outputs,epochs=10000)
    
    for inputRow in inputs:
        print(str(inputRow)+" "+str(nn.eval(inputRow)))
    
    nn.export("test.json")
    newNN = NeuralNetwork.fromFile("test.json")

    for inputRow in inputs:
        print(str(inputRow)+" "+str(nn.eval(inputRow)))
 

