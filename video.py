import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

print(tf.__version__ )

mnist = tf.keras.datasets.mnist #28x28 handwritten digits

#unpacking to training and testing variables

(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9
x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)


#scaling/normalising
#tensor values are now scaled between 0 and 1
#easier for a network to learn

model = tf.keras.models.Sequential()
#first layer is input layer
#our images are 28*28 in a multidimensional array so let's make them flat


#model.add(tf.keras.layers.Flatten())

#set density of neural network
#128 NODES
#activation function is what tells the
#neurons to 'fire' - we are using the
#sort of go-to ones
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.softmax))   #10 because dataset is numbers from 0 - 9

#we have 3 layers

#Now for parameters for the model's training
#adam is the sort of go-to optimiser
model.compile(optimizer='adam', 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy']
             )

model.fit(x_train, y_train, epochs=3)

plt.imshow(x_train[0], cmap = plt.cm.binary)

#val_loss, val_acc = model.evaluate (x_test, y_test)
#print(val_loss, val_acc)

print(x_train[0])
#above prints data from the database
#what it prints is a TENSOR
plt.show() 

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test]) 
 
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show() 