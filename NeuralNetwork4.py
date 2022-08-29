import numpy as np

class NeuralNetwork:
    def __init__(self, lr, layers):
        self.lr = lr
        self.weights = []
        self.bias = []
        
        for i in range(len(layers)-1):
            # Se inician los pesos de la capa i
            weight = np.random.randn(layers[i], layers[i+1])
            self.weights.append(weight)
            # Se inician los segos de la capa i
            bias = np.random.randn(layers[i+1], 1)
            self.bias.append(bias)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidDerivative(self, x):
        return x*(1 - x)

    def feedForward(self, x):
        layers_out = []
        s = x

        for i in range(len(self.weights)):
            weights = self.weights[i]
            bias = self.bias[i].T
            s = self.sigmoid(s.dot(weights) + bias)
            layers_out.append(s)
        
        return layers_out

    def backPropagation(self, layers_out, y, X):
        i = len(self.weights) - 1 # indice del ultimo peso
        layers_delta = {}

        # Se calculan las variaciones (deltas) de todas las capas
        while i >= 0:
            layer_out = layers_out[i]
        
            if i == len(self.weights) - 1:
                layer_error = y - layer_out # El error de la ultima capa se calcula como el resultado esperado menos lo obtenido
            else:
                # El error de las capas anteriores a la ultima se calcula como el producto escalar del delta de esta capa por la traspuesta del peso de esa capa
                # Ejemplo: El error de la capa 4 se calcula como: deltas_capa_4 * traspuesta(pesos_capa_4)
                layer_error = layers_delta[i+1].dot(self.weights[i+1].T) 
            
            # El delta se calcula como el error * derivadasigmoide de la salida de la capa
            delta = layer_error * self.sigmoidDerivative(layer_out)
            layers_delta[i] = delta                
            i = i - 1
        
        # Se aplican las variaciones (deltas) a todas las capas
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] = self.weights[i] + self.lr * X.T.dot(layers_delta[i])
            else:
                self.weights[i] = self.weights[i] + self.lr * layer_out.T.dot(layers_delta[i])

        # Se aplican las variacions (deltas) a todos los pesos
        for i in range(len(self.bias)):
            for g in layers_delta[i]:
                #print(len(g))
                #self.bias[i] = self.bias[i] + self.lr * g
                #np.reshape(num,(self.outputLayer,1))
                self.bias[i] = self.bias[i] + self.lr * np.reshape(g,(len(g),1))
        
    def train(self, x, epochs, verbose):
        if verbose:
            print('Pruebas: ')
            print(X)
            print('Prediccion inicial: ')
            self.predict(x)

        for i in range(epochs):
            o = self.feedForward(x)
            self.backPropagation(o, y, x)

        if verbose:
            print('Prediccion final: ')
            self.predict(x)

    def predict(self, x):
        layers_out = self.feedForward(x)
        layer_output = len(layers_out) - 1
        print(layers_out[layer_output])


X = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)
lr = 0.05 # El learning rate (taza de aprendizaje)
layers = [3, 16, 16, 16, 1] # Va a ser una red neurona de 3 x 8 x 8 x 8 x 1
neural_network = NeuralNetwork(lr, layers)
neural_network.train(X, 100000, True)
neural_network.predict(np.array([0,0,0]))