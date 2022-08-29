import numpy as np

class Neural_Network (object):
  def __init__(self):

    self.lr = 0.05
    # Parametros
    self.inputLayerSize = 3 # X1, X2, X3 (Capa de entrada)
    self.hiddenLayerSize = 4 # Z1, Z2, Z3, Z4 (Capa escondida)
    self.outputLayerSize = 1 # Y1 (Capa de salida)
    # Creamos pesos aleatorios para cada una de nuestras capas. Para esto creamos matrices para todos los pesos
    self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)  # 3x4 matriz for input to hidden
    self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) # 4x1 matrix for hidden layer to output
    
  def feedForward(self, input):
    self.z = np.dot(input, self.W1) # Producto escalar entre la entrada y la primera matriz de pesos 3x4
    self.z2 = self.activationSigmoid(self.z) # Aplicamos la activacion sigmoide a la salida de la primera capa
    self.z3 = np.dot(self.z2, self.W2) # Producto escalar entre la matriz de salida de la capa oculta y la matriz de pesos de la ultima capa 4x1
    o = self.activationSigmoid(self.z3) # Aplicamos la activacion sigmoide a la salida de la segunda capa
    return o

  def backwardPropagate(self, training_set, labels_set, o):
    # o_error = output error
    # o_delta = output delta

    self.o_error = labels_set - o # Calcula el error de la salida
    self.o_delta = self.o_error*self.activationSigmoidPrime(o) # Aplica la derivada de la funcion de activacion Sigmoide al error

    self.z2_error = self.o_delta.dot(self.W2.T)
    self.z2_delta = self.z2_error*self.activationSigmoidPrime(self.z2)

    # Se suman todos los pesos de los deltas
    self.W1 += training_set.T.dot(self.z2_delta)
    self.W2 += self.z2.T.dot(self.o_delta)

  def trainCicle(self, training_set, labels_set):
    o = self.feedForward(training_set)
    self.backwardPropagate(training_set, labels_set, o)

  def activationSigmoid(self, s):
    return 1/(1+np.exp(-s))

  def activationSigmoidPrime(self, s):
    return s * (1 - s)

  def train(self, training_set, labels_set, trainingEpochs = 1000, verbose = False):
    for i in range(trainingEpochs):
      self.trainCicle(training_set, labels_set)

    if (verbose):
      print ("Network Input : \n" + str(training_set))
      print ("Expected Output of XOR Gate Neural Network: \n" + str(labels_set))
      print ("Actual Output from XOR Gate Neural Network: \n" + str(self.feedForward(X)))
  
  def predict(self, input):
    print(self.feedForward([input]))


# X = input of our 3 input XOR gate
# Las pruebas de nuestra red neuronal
X = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
# El resultado de cada una de nuestras pruebas
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)
# El valor que intentamos predecir

myNeuralNetwork = Neural_Network()
myNeuralNetwork.train(X, y, 1, True)
myNeuralNetwork.predict([0,0,0])