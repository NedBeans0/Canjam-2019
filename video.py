import tensorflow as tf
import matplotlib.pyplot as plt 

print(tf.__version__ )

mnist = tf.keras.datasets.mnist #28x28 handwritten digits

#unpacking to training and testing variables

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scaling/normalising


plt.imshow(x_train[0], cmap = plt.cm.binary)

print(x_train[0])
#above prints data from the database
#what it prints is a TENSOR

plt.show() 
