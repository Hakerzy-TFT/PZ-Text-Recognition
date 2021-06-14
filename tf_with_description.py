#%matplotlib inline
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle

from google.colab import drive
drive.mount('/content/drive')

# wczytywanie danych z pliku
data = pd.read_csv(r"drive/MyDrive/A_Z Handwritten Data.csv").astype('float32')

# dzielimy dane na X oraz Y 
# x- nasze dane (obrazki)
# y - output
X = data.drop('0',axis = 1)
y = data['0']

# Train_split dzieli dane na dane testowe i dane uczące
# convertujemy 784 kolumnuy danych z pixelami w obrazki 28x28
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))

# wypisujemy dane 
print("Dane uczace: ", train_x.shape)
print("Dane testowe: ", test_x.shape)

# definiujemy słownik słów żeby wartości numeryczne  zamienić na litery
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

# Plotting the number of alphabets in the dataset...
# wyświetlamy zawartość zbioru danych na wykresie
train_yint = np.int0(y)
count = np.zeros(26, dtype='int')
for i in train_yint:
    count[i] +=1

alphabets = []
for i in word_dict.values():
    alphabets.append(i)

fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.barh(alphabets, count)

plt.xlabel("Liczba elementów ")
plt.ylabel("Litery")
plt.grid()
plt.show()

# mieszanie liter zbioru testowego (libka sk learn)
shuff = shuffle(train_x[:100])

fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()

# wyświetlamy kilka liter ze zbiru danych dla przykładu
for i in range(9):
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
plt.show()

# dostosowanie danych testowych i numerycznych w celu załadowania ich do sieci 
train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
print("Nowe wygląd danych uczących: ", train_X.shape)

test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("Nowe wygląd danych testowych ", test_X.shape)

# Konwertujemy zmienne zmienno przecinkowe
# https://keras.io/api/utils/python_utils/
# przykład: 
# >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
#>>> a = tf.constant(a, shape=[4, 4])
#>>> print(a)
#tf.Tensor(
#  [[1. 0. 0. 0.]
#   [0. 1. 0. 0.]
#   [0. 0. 1. 0.]
#   [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)

train_yOHE = tf.keras.utils.to_categorical(train_y, num_classes = 26, dtype='int')
print("NOwy wyglad danych uczących ", train_yOHE.shape)

test_yOHE = tf.keras.utils.to_categorical(test_y, num_classes = 26, dtype='int')
print("Nowy wygląd danych tesowych", test_yOHE.shape)

#####################################################################################3
# model sieci neuronowej

# https://keras.io/api/models/sequential/
# Sequential groups a linear stack of layers into a tf.keras.Model.<- z dokumentacji
# Sequential provides training and inference features on this model. <- z dokumentacji
model = Sequential()

# https://keras.io/api/models/sequential/
# Adds a layer instance on top of the layer stack.
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# 2D convolution layer (e.g. spatial convolution over images).< splot przestrzenny na obrazie
# wszystkie wartswy mają liniową funkcję aktywacji
# https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
# Flattens the input. Does not affect the batch size.
# Inherits From: Layer, Module  <- z dokumentacji

model.add(Flatten())

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# Just your regular densely-connected NN layer.
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

# https://machinelearningmastery.com/softmax-activation-function-with-python/
# he softmax, or “soft max,” mathematical function can be thought to be a probabilistic or “softer” version of the argmax function.
model.add(Dense(26,activation ="softmax"))

# Computes the crossentropy loss between the labels and predictions.
# https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
# kompilacja modelu
# definiowanie funkcji optymalizującej 

## pass optimizer by name: default parameters will be used
# https://keras.io/api/optimizers/
# Adaptive Gradient Algorithm (AdaGrad) that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).
# sieć jest ogromna więc trenujemy ją tylko przez jedną epokę

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history = model.fit(train_X, train_yOHE, epochs=1, callbacks=[reduce_lr, early_stop],  validation_data = (test_X,test_yOHE))

model.summary() # wyświetla podsumowanie modelu
model.save(r'os.path.join("./model.h5")') 3 zapisywanie

# Wyświetlanie dokładności modelu itp
print("dokładność weryfikacji:", history.history['val_accuracy'])
print("dokładność trenowania :", history.history['accuracy'])
print("strata/loss:", history.history['val_loss'])
print("strata przy danych uczących", history.history['loss'])

# # Przewidywanie liter
# pred = model.predict(test_X[:9])
# print(test_X.shape)

# # Displaying some of the test images & their predicted labels...

# fig, axes = plt.subplots(3,3, figsize=(8,9))
# axes = axes.flatten()

# for i,ax in enumerate(axes):
#     img = np.reshape(test_X[i], (28,28))
#     ax.imshow(img, cmap="Greys")
#     pred = word_dict[np.argmax(test_yOHE[i])]
#     ax.set_title("Prediction: "+pred)
#     ax.grid()

# # Prediction on external image...

# img = cv2.imread(r'b.png')
# img_copy = img.copy()

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (400,440))

# img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
# img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

# img_final = cv2.resize(img_thresh, (28,28))
# img_final =np.reshape(img_final, (1,28,28,1))


# img_pred = word_dict[np.argmax(model.predict(img_final))]

# # cv2.putText(img, "Dataflair _ _ _ ", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
# # cv2.putText(img, "Prediction: " + img_pred, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))

# print("Prediction: " + img_pred)
# plt.imshow(img)
# plt.show()

# # while (1):
# #     k = cv2.waitKey(1) & 0xFF
# #     if k == 27:
# #         break
# # cv2.destroyAllWindows()

