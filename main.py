import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import random

class xorMLP(object):
    def __init__(self, learning_rate=0.):

        self.output3 = 0
        self.output4 = 0
        self.learning_rate = learning_rate

        #Pesos
        self.W = np.matrix([[-1.0, -1.0, -1.0,  1.0,  1.0],
                            [-1.0, -1.0, -1.0,  0.2,  0.2],
                            [-1.0, -1.0, -1.0,  0.3,  0.3],
                            [-1.0, -1.0, -1.0, -1.0,  0.4],
                            [-1.0, -1.0, -1.0, -1.0, -1.0]])

        #Bias
        self.V = np.matrix([[-1.0, -1.0, -1.0,  0.5,  0.5],
                            [-1.0, -1.0, -1.0,  0.5,  0.5],
                            [-1.0, -1.0, -1.0,  0.5,  0.5],
                            [-1.0, -1.0, -1.0, -1.0,  0.5],
                            [-1.0, -1.0, -1.0, -1.0, -1.0]])

    def fit(self):
        Xxor = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        Yxor = np.matrix([[0], [1], [1], [0]])

        #print(Xxor)
        #print(Yxor)

        tolerance = 0.6
        toleranceReached = False


        #Inicializar pesos a random
        for i in range(0, 3):
            self.W[i, 3] =  random.uniform(-1, 1)
        for i in range(0, 4):
            self.W[i, 4] = random.uniform(-1, 1)


        for i in range(0,15000):

            self.V[:3,3:4] = 0
            self.V[:4, 4] = 0
            epochError = 0

            for X,Y in zip(Xxor,Yxor):

                # Calcular salida de la red
                actualOutput = (np.asarray(X).reshape(-1))

                output1 = actualOutput[0]
                output2 = actualOutput[1]

                input3 = np.matrix([[1, output1, output2]])
                output3 = self.__sigmoidalAct(np.matmul(input3, self.W[:3, 3:4]))

                input4 = np.matrix([1, output1, output2, output3])
                output4 = self.__sigmoidalAct(np.matmul(input4, self.W[:4, 4]))


                #Calcular error total
                error4 = (Y - output4)

                #Calcular gradiente de la neurona de salida
                gradient4 = self.__sigmoidalAct(output4,derivate=True) * error4


                # Calcular gradiente de la neurona oculta
                gradient3 = self.__sigmoidalAct(output3,derivate=True) * (gradient4 * self.W[3, 4])


                #Calcular variacion de los pesos ho
                self.V[0, 4] += (self.learning_rate * gradient4 * 1)
                self.V[3, 4] += (self.learning_rate * gradient4 * output3)


                #Calcular variacion de los pesos io
                self.V[1, 4] += (self.learning_rate * gradient4 * np.asarray(X).reshape(-1)[0])
                self.V[2, 4] += (self.learning_rate * gradient4 * np.asarray(X).reshape(-1)[1])


                # # Calcular variacion de los pesos ih
                self.V[0, 3] += (self.learning_rate * gradient3 * 1)
                self.V[1, 3] += (self.learning_rate * gradient3 * np.asarray(X).reshape(-1)[0])
                self.V[2, 3] += (self.learning_rate * gradient3 * np.asarray(X).reshape(-1)[1])


                epochError += error4


            # Calcular error medio
            averageError = abs(epochError / 4)

            #Actualizar pesos
            for i in range(0,3):
                self.W[i,3] = self.W[i,3] + self.V[i,3]
            for i in range(0,4):
                self.W[i,4] = self.W[i,4] + self.V[i,4]









    def predict(self, x):
        """
        x = [x1, x2]
        """
        output1 = x[0]
        output2 = x[1]

        input3 = np.matrix([[1, output1, output2]])
        hipotesis3 = np.matmul(input3, self.W[:3,3:4])
        self.output3 = self.__sigmoidalAct(hipotesis3)

        input4 = np.matrix([1, output1, output2, self.output3])
        hipotesis4 = np.matmul(input4, self.W[:4,4])
        self.output4 = self.__sigmoidalAct(hipotesis4)

        return self.umbral(self.output4)

    def __sigmoidalAct(self,h,derivate=False):
        if(derivate==True):
            return h * (1 - h)
        return (1/(1 + np.exp(-h)))


    def umbral(self,x):
        if x > 0.7:
            return 1
        elif x < 0.3:
            return 0
        else:
            return 0.5


class DeepMLP(object):
    def __init__(self, layers_size, learning_rate=0.):
        """
        Inicializa parametros de la red neuronal
        """

        # Parametros de aprendizaje
        self.learning_rate = learning_rate
        self.training_epochs = 5
        self.batch_size = 100
        self.display_step = 1

        # Parametros de la red
        self.layersSize = layers_size
        self.n_input = self.layersSize[0] # input de MNIST
        self.n_classes = self.layersSize[len(self.layersSize)-1] # Clases de MNIST (10 numeros)
        self.n_hiddenLayers = len(self.layersSize)-2 #Capas ocultas
        self.n_hidden = self.layersSize[1:-1]#Tamanio capas ocultas


        #Placeholders de entrada y salida
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])


        #Inicializar listas de matrices de pesos y bias por capa
        self.weights = []
        self.biases = []

        for i in range(len(self.layersSize)-1):
            self.weights.append( tf.Variable( tf.random_normal([self.layersSize[i],self.layersSize[i+1]]) ))

        for i in range(len(self.layersSize)-1):
            self.biases.append( tf.Variable (tf.random_normal( [self.layersSize[i+1]] ) ))

        #Instanciar el modelo
        self.modelMLP = self.MultilayerPerceptronModel(self.X,self.weights,self.biases)



    def MultilayerPerceptronModel(self, x, weights, biases):
        layersOutput = []  # Almacena salida de cada capa

        # Feedforward
        layersOutput.append(tf.add(tf.matmul(x, weights[0]), biases[0]))  # layers[0]

        for i in range(len(self.layersSize) - 2):
            layersOutput.append(tf.add(tf.matmul(layersOutput[i], weights[i + 1]), biases[i + 1]))

        # Devuelves salida de la ultima capa
        return layersOutput[-1]


    def fit(self, X):
        """
        X = entradas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datosP de entrenamiento, puede ser un batch o una sola tupla
        """
        print("\nTraining Model...")

        #Inicializas dataset MNIST
        input = X


        # Definir funcion de coste (error), optimizador, y operacion de minimizacion de la funcion de coste
        lossFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.modelMLP, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        trainOperation = optimizer.minimize(lossFunction)


        # Inicializar variables
        init = tf.global_variables_initializer()


        #Inicializar sesion
        self.sess = tf.Session()

        self.sess.as_default()
        self.sess.run(init)

        #Bucle de entrenamiento
        for epoch in range(self.training_epochs):

            avg_cost = 0.
            total_batch = int(input.train.num_examples / self.batch_size)

            #Por batch
            for i in range(total_batch):

                #Inicializar batches
                batch_x, batch_y = input.train.next_batch(self.batch_size)

                # Ejecutar optimizacion (Backpropagation minimizando la funcion de coste)
                _, c = self.sess.run([trainOperation, lossFunction], feed_dict={self.X: batch_x, self.Y: batch_y})

                # Calcular error medio
                avg_cost += c / total_batch

            # Display
            if (epoch % self.display_step == 0):
                print("Epoch:", '%04d' % (epoch + 1), "Error={:.9f}".format(avg_cost))

        print("Finished!")


    def score(self, X):

        input = X

        # Aplicar softmax al modelo
        pred = tf.nn.softmax(self.modelMLP)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

        # Calcular precision
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Accuracy:", accuracy.eval(session=self.sess, feed_dict={self.X: input.test.images, self.Y: input.test.labels}))




    def getInfo(self):
        print('Total layers = %d , HiddenLayers = %d ' % (len(self.layersSize), self.n_hiddenLayers))
        print('Input Neurons = %d , OutputNeurons = %d  ' % (self.n_input, self.n_classes))
        print('Hidden Neurons')
        print(self.n_hidden)
        #print(self.weights)
        #print(self.biases)



    def __del__(self):
        print("Close")
        self.sess.close()




if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    layersSizes = [784 ,20, 50, 10]
    deepMLP = DeepMLP(layers_size=layersSizes, learning_rate=0.005)
    deepMLP.getInfo()
    deepMLP.fit(mnist)
    deepMLP.score(mnist)
    #deepMLP.__del__()







    mlp = xorMLP(0.5)
    mlp.fit()




    #print()
    print(mlp.predict([0, 0]))
    print(mlp.predict([0, 1]))
    print(mlp.predict([1, 0]))
    print(mlp.predict([1, 1]))





