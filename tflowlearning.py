import tensorflow as tf
#it says this gives error but it doesn't 

mnist = tf.keras.datasets.mnist 
#the mnist data set is a database 
#of handwritten digits, used to train
#recognition software


#Converts the samples from integers
#to floating-point numbers.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



#Builds the tf.kersas.Sequential model
#by stacking layers

#also chooses an optimiser and a 
#loss function used for training
model = tf.keas.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


#trains and evaluates the model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test) 

#this image classifier is now trained
#to 98% accuracy on this dataset.