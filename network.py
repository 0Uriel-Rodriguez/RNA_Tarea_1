import random
import numpy as np

class Network(object): #Aqui se define la clase Network

    def __init__(self, sizes): # Define el constructor de la clase. sizes es una lista que contiene el número de neuronas en cada capa
                               # de la red neuronal, donde sizes[0] es el número de neuronas en la capa de entrada, sizes[1] es el
                               # número de neuronas en la primera capa oculta, y así sucesivamente.
        
        self.num_layers = len(sizes) # Calcula y almacena el número de capas en la red neuronal.
        self.sizes = sizes #  Almacena la lista sizes.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # Inicializa los sesgos (biases) de la red neuronal de manera aleatoria.
                                                                 # Creando una lista de matrices. Cada matriz tiene "y" filas y 1 columna, donde
                                                                 # "y" es el número de neuronas en cada capa excepto la capa de entrada. 
        self.weights = [np.random.randn(y, x)  #Al igual que en el anterior, inicializa los pesos de la red neuronal de manera aleatoria. 
                        for x, y in zip(sizes[:-1], sizes[1:])]

# Toda la parde donde definimos feedforward. Este realiza la propagación hacia adelante en la red neuronal. Toma una entrada a 
# y pasa a través de las capas de la red aplicando una función de activación (en este caso, la función sigmoide) en cada capa.    
    def feedforward(self, a): 
       
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
        

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #Define el método SGD que toma varios parámetros:

       
        training_data = list(training_data) #Convierte los datos de entrenamiento en una lista. 
        n = len(training_data) #Calcula el número total de ejemplos de entrenamiento en training_data y lo almacena en la variable n.

        if test_data: # Verifica si se proporcionaron datos de prueba.
            test_data = list(test_data)  # Si se proporcionaron datos de prueba, los convierte en una lista.
            n_test = len(test_data) # Si se proporcionaron datos de prueba, calcula el número total de ejemplos de prueba en test_data y lo almacena en la variable n_test

        for j in range(epochs): #Inicia un bucle que itera a través del número de épocas especificado
            random.shuffle(training_data) # Aleatoriza el orden de los datos de entrenamiento. 
            mini_batches = [training_data[k:k+mini_batch_size]  # Divide los datos de entrenamiento aleatorizados en minibatch.
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: #Inicia un bucle que itera a través de cada mini
                self.update_mini_batch(mini_batch, eta)  #Actualiza los pesos y sesgos de la red neuronal utilizando el minibatc actual y la tasa de aprendizaje.
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test)) 
# Imprime el número de época actual, el número de predicciones correctas en los datos de prueba y el número total de ejemplos de prueba.
            else:
                print("Epoch {} complete".format(j))

# En la parte de def update_mini_batch basicamente actualiza los pesos y biases de la red utilizando un minibatch de datos.
# Calcula el gradiente de la función de costo con respecto a los pesos y baiases y actualiza los parámetros de la red en función
# de este gradiente.
    def update_mini_batch(self, mini_batch, eta): 
       
        nabla_b = [np.zeros(b.shape) for b in self.biases] # Inicializa una lista nabla_b con gradientes de biases, pero todos los elementos se inicializan con matrices de ceros.
        nabla_w = [np.zeros(w.shape) for w in self.weights] # Inicializa una lista nabla_w con gradientes de pesos. Esta lista tiene la misma estructura que la lista (self.weights), pero todos los elementos se inicializan con matrices de ceros.
        for x, y in mini_batch: # Inicia un bucle que itera
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # Llama al método backprop para calcular los gradientes del error con respecto a (delta_nabla_b) y con respecto a (delta_nabla_w)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # Actualiza la lista.
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #  Actualiza la lista
#Definimos los pesos y los biases con las ecuaciones dadas por el optimizador usado en este caso el SGD.
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y): # Define el método backprop, que toma dos argumentos "x" y "y"
      
        nabla_b = [np.zeros(b.shape) for b in self.biases] #  Inicializa una lista nabla_b con gradientes de baiases.
        nabla_w = [np.zeros(w.shape) for w in self.weights] #  Inicializa una lista nabla_b con gradientes de pesos.
       
        activation = x # Inicializa una variable activation con la entrada "x". Esta variable se usará para mantener la activación de cada capa durante la propagación hacia adelante.
        activations = [x] # Inicializa una lista activations que contiene todas las activaciones de las capas de la red neuronal. La primera activación en la lista es la entrada x.
        zs = [] #  Inicializa una lista zs para almacenar las entradas ponderadas (z) de cada capa. Utilizado en el cálculo de gradientes.
        for b, w in zip(self.biases, self.weights): #Inicia un bucle que itera a través de los biases y pesos
            z = np.dot(w, activation)+b # Calcula la entrada ponderada z de la capa actual utilizando los pesos w, la activación anterior activation, y los sesgos b
            zs.append(z) # Agrega la entrada ponderada z a la lista zs 
            activation = sigmoid(z) #Calcula la activación de la capa actual aplicando la función de activación sigmoide a la entrada ponderada z
            activations.append(activation) #Agrega la activación de la capa actual a la lista activations.
       
        delta = self.cost_derivative(activations[-1], y) * \ #Calcula el delta de error en la capa de salida.
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta  #Almacena el gradiente de biases en la última capa (-1) en la lista nabla_b.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # Almacena el gradiente de pesos en la última capa (-1) en la lista nabla_w.
       
        for l in range(2, self.num_layers): #  Inicia un bucle que itera a través de las capas ocultas de la red neuronal
            z = zs[-l] # Obtiene la entrada ponderada z de la capa actual desde la lista zs.
            sp = sigmoid_prime(z) #Calcula el gradiente de la función de activación sigmoide en la entrada ponderada z.
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  #Calcula el nuevo delta de error para la capa actual
            nabla_b[-l] = delta # Almacena el gradiente de biases en la capa actual en la lista nabla_b.
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #Almacena el gradiente de pesos en la capa actual en la lista nabla_w
        return (nabla_b, nabla_w) #Devuelve los gradientes de sesgos y pesos calculados para todas las capas de la red.

    def evaluate(self, test_data): # Define el método evaluate, que toma un argumento test_data, que es el conjunto de datos de prueba.
      
        test_results = [(np.argmax(self.feedforward(x)), y)  #evalúa la red neuronal en cada ejemplo de test_data.
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) #Calcula el número total de predicciones correctas en el conjunto de datos de prueba.

    def cost_derivative(self, output_activations, y):   
      
        return (output_activations-y)  #Calcula la derivada de la función de costo con respecto a la salida de la red neuronal

def sigmoid(z): #Define una función llamada sigmoid que toma un argumento z.
   
    return 1.0/(1.0+np.exp(-z)) #Ecuacion vista en clase para definir la actixacion sigmoide

def sigmoid_prime(z): # Toma un argumento z. Esta función calcula la derivada de la función sigmoide en función de z
   
    return sigmoid(z)*(1-sigmoid(z))
