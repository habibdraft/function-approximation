import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

#Celsius to Fahrenheit
fig = plt.figure()
ax = plt.axes()
C = np.linspace(0, 10, 1000)
ax.plot(C, 1.8*C+32)

#Sine function approximation
x = tf.linspace(-math.pi, math.pi, 2000)
y = tf.sin(x)
plt.plot(x, y)
plt.show()

a = tf.random.uniform(shape=[]) 
b = tf.random.uniform(shape=[])
c = tf.random.uniform(shape=[])
d = tf.random.uniform(shape=[])

learning_rate = 1e-6

for i in range(2000):
    y_pred = a + b*x + c*x**2 + d*x**3 #forward pass
    loss = tf.pow(y_pred - y, 2)
    loss = tf.reduce_sum(loss).numpy() #sum of squared errors
    grad_y_pred = 2.0 * (y_pred - y) 
    
    if i % 100 == 0:
        plt.plot(x, y_pred)
        plt.show()

    grad_a = tf.reduce_sum(grad_y_pred) #gradient descent
    grad_b = tf.reduce_sum(grad_y_pred * x)
    grad_c = tf.reduce_sum(grad_y_pred * x**2)
    grad_d = tf.reduce_sum(grad_y_pred * x**3)
    
    a -= grad_a * learning_rate #update weights
    b -= grad_b * learning_rate
    c -= grad_c * learning_rate
    d -= grad_d * learning_rate

print(f'Result: y = {a.numpy()} + {b.numpy()} x + {c.numpy()} x^2 + {d.numpy()} x^3')

#a = 0.012
#b = 0.847
#c = -0.002
#d = -0.092
