import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential, Input, Model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization 
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
#####################################################################################
#Entrar en la carpeta donde se tienen las imagenes
directorio = os.path.join(os.getcwd(),'./imag/imagenes/entrenamiento')
#ir separando los subdirectorios
imgpath = directorio + os.sep

#creamos estas variables para almacenar los mapas de pixeles de las imagenes
# y los directorios
imagenes = []
directories = []
contador_directorios = []
prevRoot = ''
cant = 0
#####################################################################################
print("leyendo imgs")
#creamos un ciclo para recorrer todo el directorio
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            #leemos la imagen y la transformamos en una matriz
            image = plt.imread(filepath)
            #añadimos la matriz a la lista de imagenes
            imagenes.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                contador_directorios.append(cant)
                cant=0
#una vez agregadas las imagenes y leidos los directorios, se obtiene cuantos se contaron
contador_directorios.append(cant)
contador_directorios = contador_directorios[1:]
contador_directorios[0]=contador_directorios[0]+1 
print(f"Directorios leidos: {len(directories)}")
print(f"Imagenes en cada carpeta: {contador_directorios}")
print(f"Total de imagenes en carpetas: {sum(contador_directorios)}")

#####################################################################################
#variables para guardar las etiquetas 
labels = []
indice = 0
for cantidad in contador_directorios:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice + 1
print("Cantidad etiquetas creadas", len(labels))
#variables temporales para guardar las imagenes y agregarlas 
cerebros = []
indice = 0
#leemos los sub-directorios, cada uno sera una etiqueta
# en este caso tendremos las etiquetas Yes y No
print(f"Indice | Nombre Etiqueta")
for directorio in directories:
    name = directorio.split(os.sep)
    print(f"[{indice}] --> [{name[len(name)-1]}]")
    cerebros.append(name[len(name)-1])
    indice = indice + 1

#guardamos los array de etiquetas en la variable y
#guardamos los array de las imagenes en la variable x
y = np.array(labels)
x = np.array(imagenes, dtype=np.uint8)

classes = np.unique(y)
nClasses = len(classes)
print("Total de etiquetas", nClasses)
print("Idice etiquetas: ", classes)
#####################################################################################
#creamos sets de entrenamiento y de test (sobre el directorio de entrenamiento)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#convertimos el tipo de numpy en uno aceptado por keras
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#dividimos los pixeles en 255 para obtener rango de valores entre 0 y 1
x_train = x_train / 255
x_test = x_test / 255
#transformamos las etiquetas a salidas en rangos de 0 y 1 con one-hot encoding
train_y_onehot = np_utils.to_categorical(y_train)
test_y_onehot = np_utils.to_categorical(y_test)

print(f"Etiqueta original para no {y_train[0]}")
print(f"Conversion One-Hot para no{train_y_onehot[0]}")

x_train, x_valid, train_label, valid_label = train_test_split(x_train, train_y_onehot, test_size=0.2, random_state=13)
print(x_train.shape, x_valid.shape, train_label.shape, valid_label.shape)
#####################################################################################
#Creamos las red
# ratio de aprendizaje
RATIO = 1e-3
#epocas con las que va a entrenar
epocas = 6
#tamaño del lote para cada epoca
tam_lote = 64
#instancia del Modelo Sequential
modelo = Sequential()
#añadimos los parametros para la red convolucional
modelo.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))
modelo.add(LeakyReLU(alpha=0.1))
modelo.add(MaxPooling2D((2, 2),padding='same'))
modelo.add(Dropout(0.5))

modelo.add(Flatten())
modelo.add(Dense(32, activation='linear'))
modelo.add(LeakyReLU(alpha=0.1))
modelo.add(Dropout(0.5))
#definimos la funcion para las etiquetas
modelo.add(Dense(nClasses, activation='softmax'))

modelo.summary()
modelo.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adagrad(learning_rate=RATIO, decay=RATIO / 100),metrics=['accuracy'])

#####################################################################################
#entrenamiento del modelo
sport_train_dropout = modelo.fit(x_train, train_label, batch_size=tam_lote, epochs=epocas, verbose=1, validation_data=(x_valid, valid_label))
# se guarda el modelo para luego evitar entrenarlo de nuevo, esto es opcional
modelo.save("sports_mnist.h5py")
#####################################################################################

test_eval = modelo.evaluate(x_test, test_y_onehot, verbose=1)

print(f"Perdida en el entrenamiento (%):  {test_eval[0]}")
print(f"Precision del entrenamiento (%):  {test_eval[1]}")

#####################################################################################
#5302.jpg  -> yes
#50202 -> no

#cargar la imagen de prueba
nombre = '5302.jpg'
imagen = plt.imread(f"./imag/imagenes/test/yes/{nombre}")
tests = [imagen]
x = np.array(tests)
#respuesta = sport_model.predict_classes(x, batch_size=batch_size,verbose=1)
respuesta = np.argmax(modelo.predict(x))
print(f"Etiqueta al que pertenece la imagen {nombre}: {respuesta} ")
if respuesta == 0:
    print(f"la imagen pertenece al grupo de cerebros sin tumor\ncon una precision de: {test_eval[1]}")
if respuesta == 1:
    print(f"la imagen pertenece al grupo de cerebros con tumor\ncon una precision de: {test_eval[1]}")