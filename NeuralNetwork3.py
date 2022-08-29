import numpy as np
# Red Neuronal de 3 x 4 x 8 x 4 x 1

class NeuralNetwork:
    def __init__(self, lr):
        # Defino el learning rate
        self.lr = lr
        # Defino la cantidad de neuronas de cada capa
        self.inputLayer = 3
        self.hiddenLayerOne = 4
        self.hiddenLayerTwo = 8
        self.hiddenLayerThree = 4
        self.outputLayer = 1
        # Ahora voy a crear los pesos de cada capa de manera aleatoria
        self.W1 = np.random.randn(self.inputLayer, self.hiddenLayerOne)
        self.W2 = np.random.randn(self.hiddenLayerOne, self.hiddenLayerTwo)
        self.W3 = np.random.randn(self.hiddenLayerTwo, self.hiddenLayerThree)
        self.W4 = np.random.randn(self.hiddenLayerThree, self.outputLayer)
        # Ahora voy a crear los sesgos de cada capa
        self.B1 = np.random.randn(self.hiddenLayerOne, 1)
        self.B2 = np.random.randn(self.hiddenLayerTwo, 1)
        self.B3 = np.random.randn(self.hiddenLayerThree, 1)
        self.B4 = np.random.randn(self.outputLayer, 1)

    def feedForward(self, X):
        self.o_hiddenLayerOne = self.sigmoid(X.dot(self.W1) + self.B1.T)# salida de la primera capa oculta
        self.o_hiddenLayerTwo = self.sigmoid(self.o_hiddenLayerOne.dot(self.W2) + self.B2.T) # salida de la segunda capa oculta
        self.o_hiddenLayerThree = self.sigmoid(self.o_hiddenLayerTwo.dot(self.W3) + self.B3.T) # salida de la tercera capa oculta
        self.o_outputLayer = self.sigmoid(self.o_hiddenLayerThree.dot(self.W4) + self.B4.T)
        return self.o_outputLayer
    
    def backPropagation(self, X, y, o):
        # Voy a calcular el delta de todas las salidas
        outputLayer_error = y - o
        outputLayer_delta = outputLayer_error * self.sigmoidDerivative(o)

        hiddenLayerThree_error = outputLayer_delta.dot(self.W4.T)
        hiddenLayerThree_delta = hiddenLayerThree_error * self.sigmoidDerivative(self.o_hiddenLayerThree)

        hiddenLayerTwo_error = hiddenLayerThree_delta.dot(self.W3.T)
        hiddenLayerTwo_delta = hiddenLayerTwo_error * self.sigmoidDerivative(self.o_hiddenLayerTwo)

        hiddenLayerOne_error = hiddenLayerTwo_delta.dot(self.W2.T)
        hiddenLayerOne_delta = hiddenLayerOne_error * self.sigmoidDerivative(self.o_hiddenLayerOne)

        #Ahora que he calculado cuanto deven variar los pesos de todas las capas (deltas), hago todoas las modificaciones
        # La variacion se obtiene como: delta * pesos
        self.W4 += self.o_hiddenLayerThree.T.dot(outputLayer_delta)
        self.W3 += self.o_hiddenLayerTwo.T.dot(hiddenLayerThree_delta)
        self.W2 += self.o_hiddenLayerOne.T.dot(hiddenLayerTwo_delta)
        self.W1 += X.T.dot(hiddenLayerOne_delta)

        # Ahora modifico los sesgos
        for num in outputLayer_delta:
            self.B4 =  self.B4 + self.lr * np.reshape(num,(self.outputLayer,1))

        for num in hiddenLayerThree_delta:
            self.B3 = self.B3 + self.lr * np.reshape(num,(self.hiddenLayerThree,1))

        for num in hiddenLayerTwo_delta:
            self.B2 = self.B2 + self.lr * np.reshape(num,(self.hiddenLayerTwo,1))

        for num in hiddenLayerOne_delta:
            self.B1 = self.B1 + self.lr * np.reshape(num,(self.hiddenLayerOne,1))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidDerivative(self, x):
        return x*(1 - x)

    def train(self, epochs = 1, X = [], y = [], verbose = True):
        for epoch in range(epochs):
            o = self.feedForward(X)
            self.backPropagation(X, y, o)
        if verbose:
            self.predict(X)

    def predict(self, X):
        print(self.feedForward(X))

X = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)

neural_network = NeuralNetwork(0.5)
neural_network.train(1, X , y)
neural_network.predict(np.array([1,0,0]))
