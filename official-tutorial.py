#import numpy as np
#from PIL import Image # Python Imaging Library (PIL)
import matplotlib.pyplot as plt
import tensorflow as tf

# load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist

# convert the samples from integers to floating-point numbers
# the original value is in range [0,255]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# build the tf.keras.Sequential model by stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# plot the first image
plt.imshow(x_train[1],cmap="gray")
plt.show()


# check created network. The network is untrained now, the predictions gives 
# wrong information for predicting classes
predictions = model(x_train[:1]).numpy()
predictions

# tf.nn.softmax function converts thest "logits" to "probabilities" for each class
tf.nn.softmax(predictions).numpy()

# loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(True))

loss_fn(y_train[:1], predictions).numpy()

# train the neural network
model.compile(optimizer='adam',loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# evaluate test data
model.evaluate(x_test,y_test,verbose=2)

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])
