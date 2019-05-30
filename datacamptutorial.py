import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np #these aren't actual errors

#https://www.datacamp.com/community/tutorials/tensorflow-tutorial


#Passing two arrays of four numbers as constants.
#Since tensors are essentially array-based, we will find ourselves working with arrays often.

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8]) 

#Multiplying arrays with the tensorflow multiplication function and printing results
result = tf.multiply(x1, x2)

print(result)

#Notice result is 'Tensor("Mul:0", shape=(4,), dtype=int32)' - not really the output we might
#intuitively expect from a multiplication function
#Tensorflow has lazy evaluation 

#To see the evaluation properly, we need to set up what is called an 'interactive session'.
sess = tf.Session() 

print(sess.run(result))
#prints 5, 12, 21, 32. Each element in the first array being multiplied with the respective element in the second

#Closes session, no longer taking up active memory.
sess.close()

#The session we just made is called a 'default' session, since it took no configuration options and parameters.

#An example of a session with config arguments which allow you to add configuration options to the session:

config=tf.ConfigProto(log_device_placement=True)
#Above logs the device that is used for an operation.


