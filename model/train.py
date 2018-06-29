
"""
train.py

It is same model as provided in keras examples on github:
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

"""

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Sequential
import keras

from keras.datasets import mnist


# Load MNIST test and train datasets using keras.datasets.mnist module

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Reshape input image data for convolutional layer and scale it to range from 0. to 1.

x_train = x_train.reshape((60000, 28, 28,1)).astype('float32')/255
x_test = x_test.reshape((10000, 28, 28,1)).astype('float32')/255


# Convert output digits to one hot vector

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Construct model architecture                                                                                  

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# Train model

model.fit(x_train,y_train,batch_size=128,epochs=12,verbose=1,validation_data=(x_test,y_test))


# Evaluate model accuracy

score = model.evaluate(x_test,y_test)

print(score)


# Save model architecture to model.json

model_json = model.to_json()

with open('model.json','w') as file:
    file.write(model_json)


# Save model weights to model.h5

model.save_weights('model.h5')

print('Saved model')