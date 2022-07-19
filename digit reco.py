import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

data = tf.keras.datasets.mnist

df = data.load_data()

(x_train  , y_train),(x_test,y_test)=df

x_train_flat=x_train.reshape(len(x_train),28*28)
x_test_flat=x_test.reshape(len(x_test),28*28)


model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=3)

prediction= model.predict(x_test)

model.evaluate(x_test,y_test)
