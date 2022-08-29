import numpy as np

# Voy a crear una red neuronal de 3 x 4 x 4 x 1

class NeuralNetwork:
    def __init__(self, lr):
        # Voy a inicializar la taza de aprendizaje
        self.lr = lr

        # Voy a definir la cantidad de neuronas que va a haber en cada capa
        self.layerInput = 3
        self.layerHiddenOne = 4
        self.layerHiddenTwo = 4
        self.layerOutput = 1

        # Ahora voy a definir las matrices de pesos para cada capa
        self.W1 = np.random.randn(self.layerInput, self.layerHiddenOne)
        self.W2 = np.random.randn(self.layerHiddenOne, self.layerHiddenTwo)
        self.W3 = np.random.randn(self.layerHiddenTwo, self.layerOutput)

        # Ahora voy a definir todos los pesos
        self.B1 = np.random.randn(self.layerHiddenOne, 1)
        self.B2 = np.random.randn(self.layerHiddenTwo, 1)
        self.B3 = np.random.randn(self.layerOutput, 1)

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    
    def sigmoidDerivative(self, x):
        return x*(1 - x)

    def feedForward(self, X):
        #print(tests)
        # Salida de la primera capa oculta
        o_hiddenLayerOne = np.dot(X, self.W1)
        self.o_hiddenLayerOne = self.sigmoid(o_hiddenLayerOne)
        
        # Salida de la segunda capa oculta
        o_hiddenLayerTwo = np.dot(self.o_hiddenLayerOne, self.W2) + self.B2.T
        self.o_hiddenLayerTwo = self.sigmoid(o_hiddenLayerTwo)

        # Salida de la capa de salida
        o_outputLayer = np.dot(self.o_hiddenLayerTwo, self.W3) + self.B3.T
        self.o_outputLayer = self.sigmoid(o_outputLayer)

        return self.o_outputLayer

    def backPropagation(self, X, y, o):
        # Descripcion: Backpropagation se caracteriza por ser un proceso que va desde la ultima capa hacia la primera
        # Delta se calcula como: delta = error * sigmoid(output)

        # Calculo el error de la salida (Resultado requerido - resultado obtenido)
        self.l3_error = y - o
        self.l3_delta = self.l3_error * self.sigmoidDerivative(o) # Calculo cuando debe variar la salida para que la funcion de costo baje

        # La forma de calcular el error de las capas posteriores es diferente: se hace el producto escalar del peso por el delta de la capa posterior
        self.l2_error = self.l3_delta.dot(self.W3.T)
        self.l2_delta = self.l2_error * self.sigmoidDerivative(self.o_hiddenLayerTwo)

        self.l1_error = self.l2_delta.dot(self.W2.T)
        self.l1_delta = self.l1_error * self.sigmoidDerivative(self.o_hiddenLayerOne)

        # Ahora que ya tengo los deltas de las salidas de todas las capas puedo modificar los pesos de la red
        # Recordatorio: En backpropagation se usa la regla delta para modificar los pesos:
        #  W = W + lr * delta * entrada
        self.W3 += self.lr * self.o_hiddenLayerTwo.T.dot(self.l3_delta)
        self.W2 += self.lr * self.o_hiddenLayerOne.T.dot(self.l2_delta)
        self.W1 += self.lr * X.T.dot(self.l1_delta)

        # Ahora debo modificar los pesos. Para esto debo sumarle a los pesos los deltas de cada prueba en su respectiva capa
        for num in self.l1_delta:
            self.B1 = np.reshape(num, (4,1)) + self.B1

        for num in self.l2_delta:
            self.B2 = np.reshape(num, (4,1)) + self.B2

        for num in self.l3_delta:
            self.B3 = np.reshape(num, (1,1)) + self.B3

    def train(self, epochs = 1, verbose = True, X = [], y = []):
        if verbose:
            print('Pesos iniciales: \n')
            print(self.W1)
            print(self.W2)
            print(self.W3)
            print('Pesos iniciales: \n')
            print(self.B1)
            print(self.B2)
            print(self.B3)

        for epoch in range(epochs):
            o = self.feedForward(X)
            self.backPropagation(X, y, o)
        if verbose:
            print ("Network Input : \n" + str(X))
            print ("Actual Output from XOR Gate Neural Network: \n" + str(self.feedForward(X)))

    def predict(self, test):
        print(self.feedForward(test))


X = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)

neural_network = NeuralNetwork(0.05)
neural_network.train(epochs = 100000, verbose = False, X = X, y = y)
neural_network.predict([[1,0,1]])